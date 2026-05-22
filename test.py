import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['OMP_NUM_THREADS']='8'
from dataloader import HKGDataset
from tqdm import tqdm
from utils import calculate_ranks, metrics
import numpy as np
import argparse
import torch
import torch.nn as nn
import os
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

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)
logger.addHandler(stream_handler)

parser = argparse.ArgumentParser()
parser.add_argument('--log_name')
parser.add_argument('--exp')
parser.add_argument('--dataset_name')
parser.add_argument('--data_dir', default="./data/", type=str)
parser.add_argument('--val_size', default=4096, type=int)
parser.add_argument('--displayed_task_types', nargs='*', default=['Pri_ent', 'Pri_rel', 'Ent', 'Rel'])
parser.add_argument('--test_epoch', type=int)
parser.add_argument('--dim', default=128, type=int)
parser.add_argument('--act', default='GELU', type=str)
parser.add_argument('--num_layer', default=12, type=int)
parser.add_argument('--num_head_ent', default=8, type=int)
parser.add_argument('--num_head_rel', default=8, type=int)
parser.add_argument('--mask_eq_init', action='store_true')

args = parser.parse_args()
        
os.makedirs(f"./logs/{args.exp}/{args.dataset_name}", exist_ok=True)

file_format = args.log_name
file_handler = logging.FileHandler(f"./logs/{args.exp}/{args.dataset_name}/{file_format}_test_{args.test_epoch}.log")
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

logger.info(f"{os.getpid()}")
for arg_name in vars(args).keys():
    logger.info(f"{arg_name}:{vars(args)[arg_name]}")
logger.info("Args Listed!")

dataset = HKGDataset(
    datasets_dir=args.data_dir,
    dataset_name=args.dataset_name,
    logger=logger
)
    
model = KREPE(
    act=args.act,
    dim=args.dim,
    num_head_ent=args.num_head_ent,
    num_head_rel=args.num_head_rel,
    num_layer=args.num_layer,
    model_dropout=0.0,
    mask_eq_init = args.mask_eq_init
).cuda()

logger.info(f"# Params:{sum(p.numel() for p in model.parameters())}")
model.load_state_dict(torch.load(f"./ckpt/{args.exp}/{args.dataset_name}/{file_format}_{args.test_epoch}.ckpt")["model_state_dict"])
model.eval()

metric_lists = {"Pri_head":[], "Pri_tail":[], "Pri_ent":[], "Pri_rel":[], "Pri":[], "Qual_ent":[], "Qual_rel":[], "Qual":[], "Ent":[], "Rel":[], "All":[]}

    
with torch.no_grad():
    rank_lists = {"Pri_head":[], "Pri_tail":[], "Pri_ent":[], "Pri_rel":[], "Pri":[], "Qual_ent":[], "Qual_rel":[], "Qual":[], "Ent":[], "Rel":[], "All":[]}

    inf_graph = dataset.inference_graph
    
    emb_ents, emb_rels = model(inf_graph.pri.clone().detach(), inf_graph.qual.clone().detach(), inf_graph.qual2fact,
                               inf_graph.stats["num_ent"], inf_graph.stats["num_rel"],
                               inf_graph.hpair.clone().detach(), inf_graph.fact2hpair,
                               inf_graph.tpair.clone().detach(), inf_graph.fact2tpair,
                               inf_graph.qpair.clone().detach(), inf_graph.qual2qpair)

    for idxs in tqdm(torch.split(torch.arange(len(dataset.test_graph.query), device = "cuda"), args.val_size)):
        query_pri, query_qual, query_qual2fact, \
        query_hpair, query_fact2hpair, \
        query_tpair, query_fact2tpair, \
        query_qpair, query_qual2qpair, \
        ent_answers, rel_answers, ent_locs, rel_locs, ent_idxs, rel_idxs = dataset.eval_inputs(idxs, mode="test")
        
        ent_pred, rel_pred = model.pred(
            query_pri, query_qual, query_qual2fact,
            query_hpair, query_fact2hpair,
            query_tpair, query_fact2tpair,
            query_qpair, query_qual2qpair,
            emb_ents, emb_rels, len(ent_answers), len(rel_answers)
        )

        for i, idx in enumerate(ent_idxs):
            pred_loc = ent_locs[i]
            ranks = calculate_ranks(ent_pred[i], dataset.test_graph.answer[idx], ent_answers[i])
            
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
            ranks = calculate_ranks(rel_pred[i], dataset.test_graph.answer[idx], rel_answers[i])
            
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
            computed_metrics = np.array(list(metrics(torch.cat(rank_lists[eval_task_type], dim=0).detach().cpu().numpy())))
            metric_lists[eval_task_type].append(computed_metrics)

    logger.info(f"PERFORMANCE ON {dataset.name}")
    for eval_task_type in metric_lists:
        if eval_task_type not in args.displayed_task_types:
            continue
        if len(rank_lists[eval_task_type]) > 0:
            mr, mrr, hit10, hit3, hit1 = metric_lists[eval_task_type][-1]
            logger.info(f"Link Prediction ({eval_task_type}, {len(torch.cat(rank_lists[eval_task_type], dim=0))})\tMR:{mr:.3f}\tMRR:{mrr:.3f}\tHit1:{hit1:.3f}\tHit3:{hit3:.3f}\tHit10:{hit10:.3f}")
