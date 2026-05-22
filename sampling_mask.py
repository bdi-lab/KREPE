import re
from dataloader import HKGDataset
import importlib
from tqdm import tqdm
import numpy as np
import argparse
import torch
import torch.nn as nn
import os
import math
import random
import logging
from sampler import DiffusionSampler
import sys
sys.path.append(os.path.dirname("./baselines/"))
sys.path.append(os.path.dirname("./verification/"))
from mask_utils import get_anchor_element, load_masked_facts, load_train_facts, get_anchor_element_with_idx
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
parser.add_argument('--input_file', type=str, default="./data/WikiPeople--eval/arbitrary_masking.txt")
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--ent_p', type=float, default=0.3)
parser.add_argument('--ent_temp', type=float, default=0.3)
parser.add_argument('--rel_p', type=float, default=0.3)
parser.add_argument('--rel_temp', type=float, default=0.05)
parser.add_argument('--max_retries', type=int, default=10)
parser.add_argument('--save_interval', type=int, default=1)
parser.add_argument('--dim', default=128, type=int)
parser.add_argument('--act', default='GELU', type=str)
parser.add_argument('--num_layer', default=12, type=int)
parser.add_argument('--num_head_ent', default=8, type=int)
parser.add_argument('--num_head_rel', default=8, type=int)
parser.add_argument('--mask_eq_init', action='store_true')
args = parser.parse_args()


def initialize_query_from_masked(masked_fact, dataset, batch_size, device):
    anchor_elements = get_anchor_element(masked_fact)
    
    fact_length = len(masked_fact)
    ent_per_query = (fact_length + 1) // 2
    rel_per_query = fact_length // 2
    
    ent_mask = torch.ones(batch_size, ent_per_query, dtype=torch.bool, device=device)
    rel_mask = torch.ones(batch_size, rel_per_query, dtype=torch.bool, device=device)
    
    pri, qual, qual2fact, hpair, fact2hpair, tpair, fact2tpair, qpair, qual2qpair = \
        dataset.sample_query(batch_size, fact_length, device)
    
    for anchor_element, anchor_idx in anchor_elements:
        if anchor_idx % 2 == 0:
            anchor_id = dataset.inference_graph.ent2id.get(anchor_element)
            if anchor_id is None:
                raise ValueError(f"Unknown anchor entity: {anchor_element}")
        else:
            anchor_id = dataset.inference_graph.rel2id.get(anchor_element)
            if anchor_id is None:
                raise ValueError(f"Unknown anchor relation: {anchor_element}")
        
        for b in range(batch_size):
            if anchor_idx == 0:
                pri[b, 0] = anchor_id
                hpair[fact2hpair[b], 0] = anchor_id
                ent_mask[b, 0] = False
                
            elif anchor_idx == 1:
                pri[b, 1] = anchor_id
                hpair[fact2hpair[b], 1] = anchor_id
                tpair[fact2tpair[b], 1] = anchor_id
                rel_mask[b, 0] = False
                
            elif anchor_idx == 2:
                pri[b, 2] = anchor_id
                tpair[fact2tpair[b], 0] = anchor_id
                ent_mask[b, 1] = False
                
            else:
                qual_idx_in_fact = (anchor_idx - 3) // 2
                qual_indices = (qual2fact == b).nonzero(as_tuple=True)[0]
                
                if qual_idx_in_fact < len(qual_indices):
                    qual_idx = qual_indices[qual_idx_in_fact]
                    
                    if anchor_idx % 2 == 1:
                        qual[qual_idx, 0] = anchor_id
                        qpair[qual2qpair[qual_idx], 0] = anchor_id
                        rel_mask[b, qual_idx_in_fact + 1] = False
                    else:
                        qual[qual_idx, 1] = anchor_id
                        qpair[qual2qpair[qual_idx], 1] = anchor_id
                        ent_mask[b, qual_idx_in_fact + 2] = False
    
    return pri, qual, qual2fact, hpair, fact2hpair, tpair, fact2tpair, qpair, qual2qpair, ent_mask, rel_mask


def save_progress(output_file, generated_facts, mode='a'):
    with open(output_file, mode, encoding='utf-8') as f:
        for fact in generated_facts:
            f.write(str(fact) + '\n')
        f.flush()

        
os.makedirs(f"./logs/{args.exp}/{args.dataset_name}", exist_ok=True)

file_format = args.log_name
log_file = f"./logs/{args.exp}/{args.dataset_name}/{file_format}_test_diffusion_masked_{args.test_epoch}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

logger.info(f"PID: {os.getpid()}")
for arg_name in vars(args).keys():
    logger.info(f"{arg_name}:{vars(args)[arg_name]}")
logger.info("Args Listed!")

logger.info(f"Loading masked facts from {args.input_file}")
masked_facts = load_masked_facts(args.input_file)
logger.info(f"Loaded {len(masked_facts)} masked facts")

dataset = HKGDataset(
    datasets_dir=args.data_dir,
    dataset_name=args.dataset_name,
    logger=logger
)

dataset_dir = os.path.join(args.data_dir, args.dataset_name)
train_facts_set = load_train_facts(dataset_dir)

model = KREPE(
    act=args.act,
    dim=args.dim,
    num_head_ent=args.num_head_ent,
    num_head_rel=args.num_head_rel,
    num_layer=args.num_layer,
    model_dropout=0.0,
    mask_eq_init = args.mask_eq_init
).cuda()

logger.info(f"# Params: {sum(p.numel() for p in model.parameters())}")
model.load_state_dict(torch.load(f"./ckpt/{args.exp}/{args.dataset_name}/{file_format}_{args.test_epoch}.ckpt")["model_state_dict"])
model.eval()

if args.output_file is None:
    args.output_file = f"./{args.exp}/{args.dataset_name}/arbitrary_masking.txt"
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

batch_facts = []

with torch.no_grad():
    inf_graph = dataset.inference_graph
    
    emb_ents, emb_rels = model(
        inf_graph.pri.clone().detach(), 
        inf_graph.qual.clone().detach(), 
        inf_graph.qual2fact,
        inf_graph.stats['num_ent'], 
        inf_graph.stats['num_rel'],
        inf_graph.hpair.clone().detach(), 
        inf_graph.fact2hpair,
        inf_graph.tpair.clone().detach(), 
        inf_graph.fact2tpair,
        inf_graph.qpair.clone().detach(), 
        inf_graph.qual2qpair
    )
    
    pbar = tqdm(enumerate(masked_facts), desc="Diffusion sampling", total=len(masked_facts))
    
    for idx, masked_fact in pbar:
        generated_fact = None
        
        anchor_element, anchor_idx = get_anchor_element_with_idx(masked_fact)
        
        fact_len = len(masked_fact)
        
        for retry_attempt in range(args.max_retries):
            try:
                logger.info(f"Masked fact {idx+1}/{len(masked_facts)}, attempt {retry_attempt+1}/{args.max_retries}")
                
                pri, qual, qual2fact, hpair, fact2hpair, tpair, fact2tpair, qpair, qual2qpair, ent_mask, rel_mask = \
                    initialize_query_from_masked(masked_fact, dataset, batch_size=1, device=torch.device('cuda'))
                
                sampler = DiffusionSampler(
                    model, 
                    batch_size=1, 
                    strategy='top_p',
                    entity_para = args.ent_p,
                    relation_para=args.rel_p,
                    ent_temp = args.ent_temp,
                    rel_temp = args.rel_temp
                )
                
                pri_tensor, qual_tensor = sampler.sample(
                    args.steps, 
                    fact_len,
                    pri, qual, qual2fact,
                    hpair, fact2hpair,
                    tpair, fact2tpair,
                    qpair, qual2qpair,
                    emb_ents, emb_rels, 
                    ent_mask, rel_mask
                )
                
                h_id = pri_tensor[0, 0].item()
                r_id = pri_tensor[0, 1].item()
                t_id = pri_tensor[0, 2].item()
                
                head_name = inf_graph.id2ent[h_id]
                rel_name = inf_graph.id2rel[r_id]
                tail_name = inf_graph.id2ent[t_id]
                
                fact = [head_name, rel_name, tail_name]
                
                qual_indices = (qual2fact == 0).nonzero(as_tuple=True)[0]
                
                for qual_idx in qual_indices:
                    qr_id = qual_tensor[qual_idx, 0].item()
                    qe_id = qual_tensor[qual_idx, 1].item()
                    
                    qr_name = inf_graph.id2rel[qr_id]
                    qe_name = inf_graph.id2ent[qe_id]
                    
                    fact.extend([qr_name, qe_name])
                
                fact_tuple = tuple(fact)
                if fact_tuple in train_facts_set:
                    continue
                else:
                    generated_fact = fact
                    break
            
            except Exception as e:
                logger.error(f"Error: {e}")
                continue
        
        if generated_fact is None:
            generated_fact = masked_fact
        
        batch_facts.append(generated_fact)
        
        if len(batch_facts) >= args.save_interval:
            save_progress(args.output_file, batch_facts, mode='a')
            batch_facts = []
    
    if batch_facts:
        save_progress(args.output_file, batch_facts, mode='a')

logger.info(f"Completed generation. Output: {args.output_file}")