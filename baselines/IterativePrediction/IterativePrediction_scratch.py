import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['OMP_NUM_THREADS']='8'
sys.path.append(os.path.dirname("../../"))
sys.path.append(os.path.dirname("../"))
from dataloader import HKGDataset
from maypl import MAYPL
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

torch.set_num_threads(8)
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(0)
np.random.seed(0)

def setup_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  
    
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)
    
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    return logger

def get_next_token(logits, is_sampling, top_k=5):
    if not is_sampling:
        return torch.argmax(logits, dim=1)
    else:
        top_logits, top_indices = torch.topk(logits, k=top_k, dim=1)
        top_probs = F.softmax(top_logits, dim=1)
        sample_indices = torch.multinomial(top_probs, num_samples=1).squeeze(1)
        final_indices = torch.gather(top_indices, 1, sample_indices.unsqueeze(1)).squeeze(1)
        return final_indices


def generate_facts(log_name, exp, dataset="wd50k-eval", data_dir="../data/", repeat_generate=100, test_epoch=3000, batch_num=10, fact_len=5, is_sampling=False,\
                   dim=256, num_init_layer=4, num_head=16, num_layer=6, model_dropout=0.2):

    os.makedirs(f"./logs/{exp}/{dataset}", exist_ok=True)
    
    log_path = f"./logs/{exp}/{dataset}/{log_name}_test_{test_epoch}_iter_scratch.log"
    logger = setup_logger(log_path)
    
    dataset_obj = HKGDataset(
        datasets_dir=data_dir,
        dataset_name=dataset,
        logger=logger
    )
    
    model = MAYPL(
        dim=dim,
        num_init_layer=num_init_layer,
        num_head=num_head,
        num_layer=num_layer,
        logger=logger,
        model_dropout=model_dropout
    ).cuda()
    
    logger.info(f"# Params: {sum(p.numel() for p in model.parameters())}")
    
    ckpt_path = f"../ckpt/{exp}/{dataset}/{log_name}_{test_epoch}.ckpt"
    model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        train_graph = dataset_obj.train_graph
        emb_ents, emb_rels, init_emb_ents, init_emb_rels = model(
            train_graph.pri.clone().detach(), 
            train_graph.qual.clone().detach(), 
            train_graph.qual2fact, 
            train_graph.stats['num_ent'], 
            train_graph.stats['num_rel'],
            train_graph.hpair.clone().detach(), 
            train_graph.fact2hpair,
            train_graph.tpair.clone().detach(), 
            train_graph.fact2tpair,
            train_graph.qpair.clone().detach(), 
            train_graph.qual2qpair
        )
        
        facts = None
        for i in tqdm(range(repeat_generate), desc="Generating facts"):
            replace = None
            for loc in range(-1, fact_len):
                query_pri, query_qual, query_qual2fact, \
                    query_hpair, query_fact2hpair, \
                    query_tpair, query_fact2tpair, \
                    query_qpair, query_qual2qpair = dataset_obj.generate_and_mask_facts(
                        facts=facts, location=loc, entities_or_rels=replace, 
                        m=batch_num, n=fact_len
                    )
                ent_preds, rel_preds = model.pred(
                    query_pri, query_qual, query_qual2fact,
                    query_hpair, query_fact2hpair,
                    query_tpair, query_fact2tpair,
                    query_qpair, query_qual2qpair,
                    emb_ents, emb_rels, init_emb_ents, init_emb_rels
                )
                
                if (loc + 1) % 2 == 0:
                    replace = get_next_token(ent_preds, is_sampling)
                elif (loc + 1) % 2 == 1:
                    replace = get_next_token(rel_preds, is_sampling)
                    
                facts = query_pri.tolist()
                for q, fact_idx in zip(query_qual.tolist(), query_qual2fact.tolist()):
                    if 0 <= fact_idx < len(facts):
                        facts[fact_idx].extend(q)
                        
        for fact in facts:
            for i, j in enumerate(fact):
                if i % 2 == 0:
                    fact[i] = train_graph.id2ent[j]
                else:
                    fact[i] = train_graph.id2rel[j]
        
        logger.info(f"Generated {len(facts)} facts")
        
        return facts

def main():
    log_name = "MAYPL_for_KREPE"
    exp = "ICML2026"
    test_epoch = 3000
    data_dir = "../../data/"
    dataset = "wd50k-eval"
    configs = [
        (864, 3),
        (89, 5),
        (40, 7),
        (4, 9),
        (2, 11),
        (1, 13)
    ]
    for repeat, fact_len in configs:
        try:
            facts = generate_facts(
                log_name=log_name,
                exp=exp,
                dataset=dataset,
                data_dir=data_dir,
                repeat_generate=11,
                test_epoch=test_epoch,
                batch_num=repeat,
                fact_len=fact_len,
                is_sampling=True,
                dim=256,
                num_init_layer=4,
                num_head=16,
                num_layer=6,
                model_dropout=0.2
            )
            
            output_file = f"./created_data/{dataset}/iter_scratch_len{fact_len}.txt"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for fact in facts:
                    f.write(str(fact) + '\n')
            
            print(f"Completed: {output_file}")
            print(f"Generated {len(facts)} facts")
        except Exception as e:
            print(f"Error: {e}")

    log_name = "MAYPL_for_KREPE"
    test_epoch = 2400
    dataset = "WikiPeople"
    configs = [
        (884, 3),
        (67, 5),
        (40, 7),
        (7, 9),
        (2, 11)
    ]    
    for repeat, fact_len in configs:
        try:
            facts = generate_facts(
                log_name=log_name,
                exp=exp,
                dataset=dataset,
                data_dir=data_dir,
                repeat_generate=11,
                test_epoch=test_epoch,
                batch_num=repeat,
                fact_len=fact_len,
                is_sampling=True,
                dim=256,
                num_init_layer=3,
                num_head=32,
                num_layer=4,
                model_dropout=0.1
            )
            
            output_file = f"./created_data/{dataset}/iter_scratch_len{fact_len}.txt"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for fact in facts:
                    f.write(str(fact) + '\n')
            
            print(f"Completed: {output_file}")
            print(f"Generated {len(facts)} facts")
        except Exception as e:
            print(f"Error: {e}")

    log_name = "MAYPL_for_KREPE"
    test_epoch = 2900
    dataset = "WikiPeople--eval"
    configs = [
        (974, 3),
        (20, 5),
        (5, 7),
        (1, 9)
    ]
    for repeat, fact_len in configs:
        try:
            facts = generate_facts(
                log_name=log_name,
                exp=exp,
                dataset=dataset,
                data_dir=data_dir,
                repeat_generate=11,
                test_epoch=test_epoch,
                batch_num=repeat,
                fact_len=fact_len,
                is_sampling=True,
                dim=256,
                num_init_layer=3,
                num_head=32,
                num_layer=4,
                model_dropout=0.1
            )
            
            output_file = f"./created_data/{dataset}/iter_scratch_len{fact_len}.txt"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for fact in facts:
                    f.write(str(fact) + '\n')
            
            print(f"Completed: {output_file}")
            print(f"Generated {len(facts)} facts")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()