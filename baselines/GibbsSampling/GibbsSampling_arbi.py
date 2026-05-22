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
import json
import random
from mask_utils import get_anchor_element, load_masked_facts, load_train_facts


torch.set_num_threads(8)
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
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

def initialize_fact_from_masked(masked_fact, train_graph, anchor_elements):
    fact_len = len(masked_fact)
    fact_ids = []
    
    for idx in range(fact_len):
        is_anchor_position = False
        for anchor_element, anchor_idx in anchor_elements:
            if idx == anchor_idx:
                if idx % 2 == 0:
                    fact_ids.append(train_graph.ent2id[anchor_element])
                else:
                    fact_ids.append(train_graph.rel2id[anchor_element])
                is_anchor_position = True
                break
        
        if not is_anchor_position:
            if idx % 2 == 0:
                random_ent_id = random.randint(0, train_graph.stats['num_ent'] - 1)
                fact_ids.append(random_ent_id)
            else:
                random_rel_id = random.randint(0, train_graph.stats['num_rel'] - 1)
                fact_ids.append(random_rel_id)
    
    return fact_ids


def get_next_token(logits, is_sampling, top_k=5):
    if not is_sampling:
        return torch.argmax(logits, dim=1)
    else:
        top_logits, top_indices = torch.topk(logits, k=top_k, dim=1)
        top_probs = F.softmax(top_logits, dim=1)
        sample_indices = torch.multinomial(top_probs, num_samples=1).squeeze(1)
        final_indices = torch.gather(top_indices, 1, sample_indices.unsqueeze(1)).squeeze(1)
        return final_indices


def gibbs_sampling_one_fact(masked_fact, anchor_elements,model,\
                            dataset_obj, train_graph, emb_ents, emb_rels,\
                            init_emb_ents, init_emb_rels, num_iterations=11, \
                            is_sampling=True, batch_num=1):
    fact_len = len(masked_fact)
    
    anchor_positions = {anchor_idx for _, anchor_idx in anchor_elements}

    current_fact = initialize_fact_from_masked(
        masked_fact, train_graph, anchor_elements
    )
    
    for i in range(num_iterations):
        for loc in range(fact_len):
            if loc in anchor_positions:
                continue
            
            facts = [current_fact.copy()]
            
            if loc % 2 == 0:
                facts[0][loc] = train_graph.stats['num_ent']
            else:
                facts[0][loc] = train_graph.stats['num_rel']
            
            if loc == 0:
                mask_location = -1
                replace = None
            else:
                mask_location = loc - 1
                replace = [current_fact[loc - 1]]
            
            query_pri, query_qual, query_qual2fact, \
                query_hpair, query_fact2hpair, \
                query_tpair, query_fact2tpair, \
                query_qpair, query_qual2qpair = dataset_obj.generate_and_mask_facts(
                    facts=facts, location=mask_location, entities_or_rels=replace,
                    m=batch_num, n=fact_len
                )
            
            ent_preds, rel_preds = model.pred(
                query_pri, query_qual, query_qual2fact,
                query_hpair, query_fact2hpair,
                query_tpair, query_fact2tpair,
                query_qpair, query_qual2qpair,
                emb_ents, emb_rels, init_emb_ents, init_emb_rels
            )
            
            if loc % 2 == 0:
                replace = get_next_token(ent_preds, is_sampling, top_k=5)
            else:
                replace = get_next_token(rel_preds, is_sampling, top_k=5)
            current_fact[loc] = replace[0].item() if torch.is_tensor(replace) else replace[0]
    
    return current_fact


def save_progress(output_file, generated_facts, mode='a'):
    with open(output_file, mode, encoding='utf-8') as f:
        for fact in generated_facts:
            f.write(str(fact) + '\n')
        f.flush()


def generate_with_gibbs(log_name="", exp="maypl", dataset="wd50k-eval", data_dir="../data/",\
                        input_file=None, test_epoch=3000, batch_num=1, num_iterations=11, is_sampling=True, output_file=None, save_interval=5, max_retries=3,\
                        dim=256, num_init_layer=4, num_head=16, num_layer=6, model_dropout=0.2):
    
    os.makedirs(f"./logs/{exp}/{dataset}", exist_ok=True)
    
    log_path = f"./logs/{exp}/{dataset}/{log_name}_test_{test_epoch}_gibbs_arbi.log"
    logger = setup_logger(log_path)  
    
    if output_file is None:
        output_file = f"./created_data/{dataset}/gibbs_arbi.txt"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    masked_facts = load_masked_facts(input_file)
    
    
    
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
    
    dataset_dir = os.path.join(data_dir, dataset)
    train_facts_set = load_train_facts(dataset_dir)
    
    batch_facts = []
    
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
        
        pbar = tqdm(enumerate(masked_facts), desc="Gibbs sampling", total=len(masked_facts))
        
        for idx, masked_fact in pbar:
            generated_fact = None
            
            anchor_elements = get_anchor_element(masked_fact)
            
            if not anchor_elements:
                generated_fact = masked_fact
                batch_facts.append(generated_fact)
                continue
            
            for r in range(max_retries):
                try:
                    final_fact_ids = gibbs_sampling_one_fact(
                        masked_fact,
                        anchor_elements,
                        model,
                        dataset_obj,
                        train_graph,
                        emb_ents,
                        emb_rels,
                        init_emb_ents,
                        init_emb_rels,
                        num_iterations=num_iterations,
                        is_sampling=is_sampling,
                        batch_num=batch_num
                    )
                    
                    fact = []
                    for idx_elem, j in enumerate(final_fact_ids):
                        if idx_elem % 2 == 0:
                            fact.append(train_graph.id2ent[j])
                        else:
                            fact.append(train_graph.id2rel[j])
                    
                    fact_tuple = tuple(fact)
                    if fact_tuple in train_facts_set:
                        continue
                    else:
                        generated_fact = fact
                        break
                
                except Exception as e:
                    continue
            
            if generated_fact is None:
                generated_fact = masked_fact
            
            batch_facts.append(generated_fact)
            
            if len(batch_facts) >= save_interval:
                save_progress(output_file, batch_facts, mode='a')
                batch_facts = []
        
        if batch_facts:
            save_progress(output_file, batch_facts, mode='a')
        
        return output_file


def main():
    log_name = "MAYPL_for_KREPE"
    exp = "ICML2026"
    test_epoch = 3000
    data_dir = "../../data/"
    dataset = "wd50k-eval"
    
    input_file = "../../data/wd50k-eval/arbitrary_masking.txt"
    output_file = f"./created_data/{dataset}/gibbs_arbi.txt"
    
    try:
        result = generate_with_gibbs(
            log_name=log_name,
            exp=exp,
            dataset=dataset,
            data_dir=data_dir,
            input_file=input_file,
            test_epoch=test_epoch,
            batch_num=1,
            num_iterations=11,  # 10 burn-in + 1 final
            is_sampling=True,
            output_file=output_file,
            save_interval=1,
            max_retries=10,
            dim=256,
            num_init_layer=4,
            num_head=16,
            num_layer=6,
            model_dropout=0.2
        )
        print(f"Completed: {result}")
    except Exception as e:
        print(f"Error: {e}")

    log_name = "MAYPL_for_KREPE"
    test_epoch = 2400
    dataset = "WikiPeople"
    
    input_file = "../../data/WikiPeople/arbitrary_masking.txt"
    output_file = f"./created_data/{dataset}/gibbs_arbi.txt"
    
    try:
        result = generate_with_gibbs(
            log_name=log_name,
            exp=exp,
            dataset=dataset,
            data_dir=data_dir,
            input_file=input_file,
            test_epoch=test_epoch,
            batch_num=1,
            num_iterations=11,  # 10 burn-in + 1 final
            is_sampling=True,
            output_file=output_file,
            save_interval=1,
            max_retries=10,
            dim=256,
            num_init_layer=3,
            num_head=32,
            num_layer=4,
            model_dropout=0.1
        )
        print(f"Completed: {result}")
    except Exception as e:
        print(f"Error: {e}")

    log_name = "MAYPL_for_KREPE"
    test_epoch = 2900
    dataset = "WikiPeople--eval"
    
    input_file = "../../data/WikiPeople--eval/arbitrary_masking.txt"
    output_file = f"./created_data/{dataset}/gibbs_arbi.txt"
    
    try:
        result = generate_with_gibbs(
            log_name=log_name,
            exp=exp,
            dataset=dataset,
            data_dir=data_dir,
            input_file=input_file,
            test_epoch=test_epoch,
            batch_num=1,
            num_iterations=11,  # 10 burn-in + 1 final
            is_sampling=True,
            output_file=output_file,
            save_interval=1,
            max_retries=10,
            dim=256,
            num_init_layer=3,
            num_head=32,
            num_layer=4,
            model_dropout=0.1
        )
        print(f"Completed: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()