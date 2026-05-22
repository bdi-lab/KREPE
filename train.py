import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['OMP_NUM_THREADS']='8'

from dataloader import HKGDataset
from tqdm import tqdm
from utils import calculate_ranks, metrics
from losses import kDCE_loss
import numpy as np
import argparse
import torch
import torch.nn as nn
import datetime
import random
import logging
from model import KREPE

torch.set_num_threads(8)
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def train(args, logger):
    for arg_name in vars(args).keys():
        logger.info(f"{arg_name}:{vars(args)[arg_name]}")
    logger.info("Args Listed!")
    
    dataset = HKGDataset(
        datasets_dir=args.data_dir,
        dataset_name=args.dataset_name,
        logger=logger
    )

    model = KREPE(
        act = args.act,
        dim = args.dim,
        num_layer = args.num_layer,
        num_head_ent = args.num_head_ent,
        num_head_rel = args.num_head_rel,
        model_dropout = args.model_dropout,
        mask_eq_init = args.mask_eq_init
    ).cuda()


    no_decay = ["bias", 'alpha', 'beta', 'gamma', 'log', 'mask', 'mean', 'norm']
    if not args.wd_mlp_ln:
        no_decay += ['ln']
    optimizer = torch.optim.AdamW([{"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": args.weight_decay, "lr":args.lr_max},
                                   {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0, "lr":args.lr_max}])
    scaler = torch.amp.GradScaler('cuda')

    if args.scheduler == "CosLRLinearWarmupRestart":
        LinearWarmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: max(1e-6/args.lr_max, epoch/(args.warmup_epoch-1)))
        CosLRRestart = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.restart_epoch, T_mult = args.restart_mult, eta_min = args.lr_min)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers = [LinearWarmup, CosLRRestart], milestones = [args.warmup_epoch])
    elif args.scheduler == "None":
        scheduler = None
    else:
        raise NotImplementedError

    if args.start_epoch != 0:
        model.load_state_dict(torch.load(f"./ckpt/{args.exp}/{args.dataset_name}/{file_format}_{args.start_epoch}.ckpt")["model_state_dict"])
        optimizer.load_state_dict(torch.load(f"./ckpt/{args.exp}/{args.dataset_name}/{file_format}_{args.start_epoch}.ckpt")["optimizer_state_dict"])

    best_valid_mrr = 0
    best_valid_epoch = 0
    
    for epoch in range(args.start_epoch, args.num_epoch):
        epoch_loss = 0
        
        train_graph_idxs = (torch.rand(dataset.train_graph.num_fact, device = "cuda") < args.train_graph_ratio).nonzero(as_tuple=True)[0]
        base_pri, base_qual, base_qual2fact, num_base_ents, num_base_rels, \
        base_hpair, base_fact2hpair, \
        base_tpair, base_fact2tpair, \
        base_qpair, base_qual2qpair, \
        conv_ent, conv_rel, query_idxs = dataset.train_split(train_graph_idxs)
        batch_num = len(torch.split(query_idxs, args.batch_size))
        for batch in tqdm(torch.split(query_idxs, args.batch_size)):
            optimizer.zero_grad(set_to_none = True)
            with torch.amp.autocast('cuda'):

                query_pri, query_qual, query_qual2fact, \
                query_hpair, query_fact2hpair, \
                query_tpair, query_fact2tpair, \
                query_qpair, query_qual2qpair, ent_answers, rel_answers, k_per_fact, pred_idxs, pred_locs, fact_length = dataset.train_preds(conv_ent, conv_rel, batch)
                
                emb_ents, emb_rels = model(base_pri, base_qual, base_qual2fact, \
                                           num_base_ents, num_base_rels, \
                                           base_hpair, base_fact2hpair, \
                                           base_tpair, base_fact2tpair, \
                                           base_qpair, base_qual2qpair)
                
                ent_preds, rel_preds = model.pred(query_pri, query_qual, query_qual2fact, \
                                                  query_hpair, query_fact2hpair, \
                                                  query_tpair, query_fact2tpair, \
                                                  query_qpair, query_qual2qpair, \
                                                  emb_ents, emb_rels, len(ent_answers), len(rel_answers))
                
                ent_pred_idxs = pred_idxs[pred_locs % 2 == 0]
                rel_pred_idxs = pred_idxs[pred_locs % 2 == 1]
                ent_loss = kDCE_loss(k_per_fact, ent_preds, ent_pred_idxs, ent_answers, fact_length)
                rel_loss = kDCE_loss(k_per_fact, rel_preds, rel_pred_idxs, rel_answers, fact_length)
                loss = ent_loss + rel_loss
                epoch_loss += loss.detach()
            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

        
        logger.info(f"Epoch {epoch+1} GPU:{torch.cuda.max_memory_allocated()} Loss:{epoch_loss.item()/batch_num:.6f}")
        if scheduler is not None:
            scheduler.step()

        if torch.isnan(epoch_loss):
            break

        if (epoch+1) % args.val_dur == 0:
            model.eval()
            with torch.no_grad():
                metric_lists = {"Pri_head":[], "Pri_tail":[], "Pri_ent":[], "Pri_rel":[], "Pri":[], "Qual_ent":[], "Qual_rel":[], "Qual":[], "Ent":[], "Rel":[], "All":[]}
                rank_lists = {"Pri_head":[], "Pri_tail":[], "Pri_ent":[], "Pri_rel":[], "Pri":[], "Qual_ent":[], "Qual_rel":[], "Qual":[], "Ent":[], "Rel":[], "All":[]}

                inf_graph = dataset.inference_graph
                
                emb_ents, emb_rels = model(inf_graph.pri.clone().detach(), inf_graph.qual.clone().detach(), inf_graph.qual2fact,
                                                        inf_graph.stats["num_ent"], inf_graph.stats["num_rel"],
                                                        inf_graph.hpair.clone().detach(), inf_graph.fact2hpair,
                                                        inf_graph.tpair.clone().detach(), inf_graph.fact2tpair,
                                                        inf_graph.qpair.clone().detach(), inf_graph.qual2qpair)

                for idxs in tqdm(torch.split(torch.arange(len(dataset.valid_graph.query), device = "cuda"), args.val_size)):
                    query_pri, query_qual, query_qual2fact, \
                    query_hpair, query_fact2hpair, \
                    query_tpair, query_fact2tpair, \
                    query_qpair, query_qual2qpair, \
                    ent_answers, rel_answers, ent_locs, rel_locs, ent_idxs, rel_idxs = dataset.eval_inputs(idxs, mode="valid")
                    
                    ent_pred, rel_pred = model.pred(query_pri, query_qual, query_qual2fact, \
                                                    query_hpair, query_fact2hpair, \
                                                    query_tpair, query_fact2tpair, \
                                                    query_qpair, query_qual2qpair, \
                                                    emb_ents, emb_rels, len(ent_answers), len(rel_answers))
                    
                    for i, idx in enumerate(ent_idxs):
                        pred_loc = ent_locs[i]
                        ranks = calculate_ranks(ent_pred[i], dataset.valid_graph.answer[idx], ent_answers[i])
                        
                        if pred_loc == 0:
                            rank_lists["Pri_head"].append(ranks)
                            rank_lists["Pri_ent"].append(ranks)
                            rank_lists["Pri"].append(ranks)
                        elif pred_loc == 2:
                            rank_lists["Pri_tail"].append(ranks)
                            rank_lists["Pri_ent"].append(ranks)
                            rank_lists["Pri"].append(ranks)
                        else:
                            rank_lists["Qual_ent"].append(ranks)
                            rank_lists["Qual"].append(ranks)
                        rank_lists["Ent"].append(ranks)
                        rank_lists["All"].append(ranks)
                    
                    for i, idx in enumerate(rel_idxs):
                        pred_loc = rel_locs[i]
                        ranks = calculate_ranks(rel_pred[i], dataset.valid_graph.answer[idx], rel_answers[i])
                        
                        if pred_loc == 1:
                            rank_lists["Pri_rel"].append(ranks)
                            rank_lists["Pri"].append(ranks)
                        else:
                            rank_lists["Qual_rel"].append(ranks)
                            rank_lists["Qual"].append(ranks)
                        rank_lists["Rel"].append(ranks)
                        rank_lists["All"].append(ranks)
                
                for eval_task_type in metric_lists:
                    if len(rank_lists[eval_task_type]) > 0:
                        metric_lists[eval_task_type].append(
                            np.array(list(metrics(torch.cat(rank_lists[eval_task_type], dim=0).detach().cpu().numpy())))
                        )

                for eval_task_type in metric_lists:
                    if len(metric_lists[eval_task_type]) > 0:
                        mr, mrr, hit10, hit3, hit1 = np.stack(metric_lists[eval_task_type], axis=0).mean(axis=0)
                        logger.info(f"Link Prediction ({eval_task_type})\nMR:{mr}\nMRR:{mrr}\nHit1:{hit1}\nHit3:{hit3}\nHit10:{hit10}")
                        if eval_task_type == "All":
                            all_mrr = mrr

                if not args.no_write:
                    torch.save(
                        {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        f"./ckpt/{args.exp}/{args.dataset_name}/{file_format}_{epoch+1}.ckpt"
                    )
                
                if all_mrr > best_valid_mrr:
                    best_valid_mrr = all_mrr
                    best_valid_epoch = epoch
                elif args.early_stop > 0:
                    if epoch - best_valid_epoch >= args.early_stop:
                        break
            model.train()

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default="WikiPeople--eval", type=str)
    parser.add_argument('--data_dir', default="./data/", type=str)
    parser.add_argument('--exp', default="ICML2026", type=str)
    parser.add_argument('--log_name', default=None, type=str)
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--act', default='GELU', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_epoch', default=2000, type=int)
    parser.add_argument('--val_dur', default=50, type=int)
    parser.add_argument('--val_size', default=4096, type=int)
    parser.add_argument('--num_layer', default=12, type=int)
    parser.add_argument('--num_head_ent', default=8, type=int)
    parser.add_argument('--num_head_rel', default=8, type=int)
    parser.add_argument('--early_stop', default=0, type=int)
    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--model_dropout', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--no_write', action='store_true')
    parser.add_argument('--scheduler', default = "CosLRLinearWarmupRestart", type = str)
    parser.add_argument('--lr_max', default=5e-4, type = float)
    parser.add_argument('--lr_min', default = 1e-5, type = float)
    parser.add_argument('--train_graph_ratio', default=0.7, type=float)
    parser.add_argument('--warmup_epoch', default = 200, type = int)
    parser.add_argument('--restart_epoch', default = 1800, type = int)
    parser.add_argument('--restart_mult', default = 1, type = int)
    parser.add_argument('--mask_eq_init', action='store_true')
    parser.add_argument('--wd_mlp_ln', action='store_true')

    args = parser.parse_args()

    if args.log_name is None:
        file_format = datetime.datetime.now()
    else:
        file_format = args.log_name

    if not args.no_write:
        os.makedirs(f"./ckpt/{args.exp}/{args.dataset_name}", exist_ok=True)
        os.makedirs(f"./logs/{args.exp}/{args.dataset_name}", exist_ok=True)
    else:
        file_format = None

    if not args.no_write:
        file_handler = logging.FileHandler(f"./logs/{args.exp}/{args.dataset_name}/{file_format}.log")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    logger.info(f"{os.getpid()}")

    try:
        train(args, logger)
    except Exception as e:
        logging.critical(e, exc_info=True)

    logger.info("END")