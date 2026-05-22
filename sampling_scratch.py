from json import load
import re
from dataloader import HKGDataset
from tqdm import tqdm
import numpy as np
import argparse
import torch
import torch.nn as nn
import os
import random
import logging
from sampler import DiffusionSampler
import sys
sys.path.append(os.path.dirname("./verification/"))
sys.path.append(os.path.dirname("./baselines/"))
from model import KREPE

os.environ['OMP_NUM_THREADS']='8'
torch.set_num_threads(8)
torch.cuda.empty_cache()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only = True)
os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"

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
parser.add_argument('--test_epoch', type=int)
parser.add_argument('--data_dir', default="./data/", type=str)
parser.add_argument('--fact_len', type=int, default=3)
parser.add_argument('--fact_num', type=int, default=512)
parser.add_argument('--steps', type=int, default="1000")
parser.add_argument('--ent_p', type=float, default=0.3)
parser.add_argument('--ent_temp', type=float, default=0.3)
parser.add_argument('--rel_p', type=float, default=0.3)
parser.add_argument('--rel_temp', type=float, default=0.3)
parser.add_argument('--dim', default=128, type=int)
parser.add_argument('--act', default='GELU', type=str)
parser.add_argument('--num_layer', default=12, type=int)
parser.add_argument('--num_head_ent', default=8, type=int)
parser.add_argument('--num_head_rel', default=8, type=int)
parser.add_argument('--mask_eq_init', action='store_true')
args = parser.parse_args()
        
os.makedirs(f"./logs/{args.exp}/{args.dataset_name}", exist_ok = True)

file_format = args.log_name
file_handler = logging.FileHandler(f"./logs/{args.exp}/{args.dataset_name}/{file_format}_test_diffusion_scratch_{args.test_epoch}.log")
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


with torch.no_grad():
    sampler = DiffusionSampler(model, args.fact_num, 'top_p', args.ent_p, args.ent_temp, args.rel_p, args.rel_temp)
    generated = []
    pri, qual, qual2fact,\
        hpair, fact2hpair,\
        tpair, fact2tpair,\
        qpair, qual2qpair = dataset.sample_query(args.fact_num, args.fact_len)
    inf_graph = dataset.inference_graph

    emb_ents, emb_rels = model(
        inf_graph.pri.clone().detach(), inf_graph.qual.clone().detach(), inf_graph.qual2fact,
        inf_graph.stats['num_ent'], inf_graph.stats['num_rel'],
        inf_graph.hpair.clone().detach(), inf_graph.fact2hpair,
        inf_graph.tpair.clone().detach(), inf_graph.fact2tpair,
        inf_graph.qpair.clone().detach(), inf_graph.qual2qpair
    )
    pri_tensor, qual_tensor = sampler.sample(args.steps, args.fact_len, \
                    pri, qual, qual2fact, \
                    hpair, fact2hpair, \
                    tpair, fact2tpair, \
                    qpair, qual2qpair, \
                    emb_ents, emb_rels
                )
    batch_size = pri_tensor.shape[0]

    for b in range(batch_size):
        h_id = pri_tensor[b, 0].item()
        r_id = pri_tensor[b, 1].item()
        t_id = pri_tensor[b, 2].item()
        
        head_name = inf_graph.id2ent[h_id]
        rel_name = inf_graph.id2rel[r_id]
        tail_name = inf_graph.id2ent[t_id]

        primary_ori = [head_name, rel_name, tail_name]
        
        qual_indices = (qual2fact == b).nonzero(as_tuple=True)[0]
        
        qual_ori = []
        for qual_idx in qual_indices:
            qr_id = qual_tensor[qual_idx, 0].item()
            qe_id = qual_tensor[qual_idx, 1].item()
                        
            qual_ori.extend([inf_graph.id2rel[qr_id], inf_graph.id2ent[qe_id]])

        ori_list = primary_ori
        if qual_ori:
            ori_list.extend(qual_ori)

        generated.append(ori_list)  

    output_dir = f"./{args.exp}/{args.dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/scratch_{args.fact_len}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        
        for fact in generated:
            f.write(str(fact)+'\n')
