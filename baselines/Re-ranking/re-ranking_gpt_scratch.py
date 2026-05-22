import os
import json
import random
import logging
from collections import defaultdict
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from openai import OpenAI
import sys
sys.path.append(os.path.dirname("../../"))
sys.path.append(os.path.dirname("../"))
sys.path.append(os.path.dirname("../../verification/"))
from dataloader import HKGDataset
from converting import convert_to_str, load_data
from maypl import MAYPL

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENAI_API_KEY'] = 'API_KEY_HERE' # Replace with your actual API key

torch.set_num_threads(8)
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(0)
random.seed(0)
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


def safe_sample(population, k):
    if not population:
        return []
    actual_k = min(len(population), k)
    return random.sample(population, actual_k)


def load_facts(dataset_dir):
    facts = []
    for name in ["train.txt"]:
        path = os.path.join(dataset_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                for line in f.readlines():
                    elements = line.strip().split()
                    facts.append(elements)
    return facts


def build_indices(facts):
    entity_index = defaultdict(list)
    relation_index = defaultdict(list)
    
    for fact in facts:
        if len(fact) < 3: continue
        h, r, t = fact[0], fact[1], fact[2]
        entity_index[h].append(fact)
        entity_index[t].append(fact)
        relation_index[r].append(fact)
        
        if len(fact) > 3:
            for i in range(3, len(fact), 2):
                if i+1 < len(fact):
                    q_rel = fact[i]
                    q_val = fact[i + 1]
                    relation_index[q_rel].append(fact)
                    entity_index[q_val].append(fact)
    
    return entity_index, relation_index


def generate_examples(statements, entity_index, relation_index, num_example, dataset_name):
    statement_str = convert_to_str(statements, dataset_name)
    
    full = []
    for loc in range(len(statements)):
        if loc % 2 == 0:
            full.append((loc, safe_sample(entity_index[statements[loc]], num_example)))
        else:
            full.append((loc, safe_sample(relation_index[statements[loc]], num_example)))
    
    analogy = []
    supplement = []
    
    for loc, statement in full:
        for fact in statement:
            if loc < len(fact):
                ans = fact[loc]
                fact_copy = fact.copy()
                fact_copy[loc] = "[MASK]"
                st = convert_to_str(fact_copy, dataset_name)
                ans_str = convert_to_str([ans], dataset_name, i=loc)
                
                if st != statement_str:
                    if loc % 2 == 0:
                        supplement.append((st, ans_str))
                    else:
                        analogy.append((st, ans_str))
    
    return analogy, supplement


def generate_prompt_first(statements, num_example, dataset_name="wd50k-eval", data_dir="../data/"):
    dataset_dir = os.path.join(data_dir, dataset_name)
    facts = load_facts(dataset_dir)
    entity_index, relation_index = build_indices(facts)
    analogy, supplement = generate_examples(statements, entity_index, relation_index, num_example, dataset_name)
    
    prompt1 = """
    You are an expert for Knowledge Graph Completion tasks.
    Your goal is to perform link prediction. This involves filling in a missing element (denoted as [MASK]) in a hyper relational fact.
    The missing element could be an Entity or a Relation.
    Given a goal statement with a [MASK] and a list of candidate answers, you need to rank the candidates based on plausibility.
    If you understand your responsibility, respond "Yes". Otherwise, respond "No". Do not output anything except "Yes" and "No".
    """

    prompt2 = f"""
    To sort the candidate answers, you need to refer to other examples that may be similar or related to it.
    Some of the given examples are similar to the goal statement. You should draw analogies from them to understand the potential meaning of the goal statement.
    Other provided facts contain supplementary information; capture this extra information and mine potential relationships among them to help the sorting.
    Please carefully read, analyze, and reflect on these examples. Identify the reasoning patterns demonstrated in these examples and retain any information that may help your verification task.
    While I provide examples, please remain silent until I ask you to respond.
    """

    prompt3 = "Examples used for analogy: "
    for st, ans in analogy:
        prompt3 += f""" Predict the [MASK] from the given "{st}". The answer is {ans}, so the [MASK] is {ans}."""
    prompt3 += " Examples used to supplement information: "
    for st, ans in supplement:
        prompt3 += f""" Predict the [MASK] from the given "{st}". The answer is {ans}, so the [MASK] is {ans}."""
    prompt3 += " Keep thinking, but DO NOT give me any feedback."
    
    return prompt1, prompt2, prompt3


def generate_prompt_question(statements, candidate_answer, dataset_name="wd50k-eval"):
    """Generate question prompt for ranking candidates"""
    statement_str = convert_to_str(statements, dataset_name)
    candidate_answer_string = " | ".join(candidate_answer)
    
    prompt4 = f"""
    The list of candidate answers is {candidate_answer_string} and the question is predict the [MASK] from the given "{statement_str}".
    The goal is to verify the fact "{statement_str}". Based on the previous examples and your own knowledge, determine the single most probable answer from the candidate list.
    Output ONLY the index of the best candidate based on the original list order."""
    
    return prompt4


def call_openai_with_retry(client, model_name, messages, max_retries=5, logger=None):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=1.0
            )
            return response
        
        except Exception as e:
            error_str = str(e).lower()
            
            if 'rate limit' in error_str or 'quota' in error_str or '429' in error_str:
                wait_time = min(60 * (2 ** attempt), 300)
                if logger:
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                time.sleep(wait_time)
                
            elif '500' in error_str or '503' in error_str:
                wait_time = 10 * (attempt + 1)
                if logger:
                    logger.warning(f"Server error. Waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                time.sleep(wait_time)
                
            else:
                if logger:
                    logger.error(f"OpenAI API Error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise
    
    raise Exception(f"Failed after {max_retries} retries")


def save_progress(output_file, generated_facts, mode='a'):
    with open(output_file, mode, encoding='utf-8') as f:
        for fact in generated_facts:
            f.write(str(fact) + '\n')
        f.flush()


def generate_with_openai(
    log_name="",
    exp="maypl",
    dataset="wd50k-eval",
    data_dir="../data/",
    repeat_generate=1,
    test_epoch=3000,
    batch_num=1,
    fact_len=5,
    num_example=1,
    top_k=5,
    model_name="gpt-5.2",
    output_file=None,
    save_interval=10,
    dim=256, num_init_layer=4, num_head=16, num_layer=6, model_dropout=0.2
):
    os.makedirs(f"./logs/{exp}/{dataset}", exist_ok=True)
    
    log_path = f"./logs/{exp}/{dataset}/{log_name}_test_{test_epoch}_gpt_scratch_len{fact_len}.log"
    logger = setup_logger(log_path)
    
    if output_file is None:
        output_file = f"./created_data/{dataset}/re-ranking_gpt_scratch_len{fact_len}.txt"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    completed_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            completed_count = sum(1 for line in f if line.strip())
        if completed_count > 0:
            if completed_count >= repeat_generate:
                return output_file
    
    data_mape = load_data(f"{data_dir}{dataset}/entities_labels.txt")
    data_mapr = load_data(f"{data_dir}{dataset}/relations_labels.txt")
    
    
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
    
    client = OpenAI()
    
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
        
        pbar = tqdm(range(completed_count, repeat_generate), desc=f"Generating (len={fact_len})", initial=completed_count, total=repeat_generate)
        
        for i in pbar:
            try:
                replace = None
                filtered_facts = [x for x in train_graph.facts if len(x) == fact_len]
                    
                facts = [random.choice(filtered_facts)]
                
                facts_id = []
                for idx, ele in enumerate(facts[0]):
                    if idx % 2 == 0:
                        facts_id.append(train_graph.id2ent[ele])
                    else:
                        facts_id.append(train_graph.id2rel[ele])
                
                prompt1, prompt2, prompt3 = generate_prompt_first(
                    facts_id, num_example=num_example, dataset_name=dataset, data_dir=data_dir
                )
                
                messages = [
                    {"role": "user", "content": prompt1},
                    {"role": "assistant", "content": "Yes"},
                    {"role": "user", "content": prompt2},
                    {"role": "assistant", "content": "I have received your instructions and the goal statement"},
                    {"role": "user", "content": prompt3},
                    {"role": "assistant", "content": "I have analyzed the examples provided and will keep thinking without giving feedback."},
                ]
                
                for loc in range(-1, fact_len):
                    query_pri, query_qual, query_qual2fact, \
                        query_hpair, query_fact2hpair, \
                        query_tpair, query_fact2tpair, \
                        query_qpair, query_qual2qpair = dataset_obj.generate_and_mask_facts(
                            facts=facts, location=loc, entities_or_rels=replace, m=batch_num, n=fact_len
                        )
                    
                    if loc == fact_len - 1:
                        facts = query_pri.tolist()
                        for q, fact_idx in zip(query_qual.tolist(), query_qual2fact.tolist()):
                            if 0 <= fact_idx < len(facts):
                                facts[fact_idx].extend(q)
                        break
                    
                    ent_preds, rel_preds = model.pred(
                        query_pri, query_qual, query_qual2fact,
                        query_hpair, query_fact2hpair,
                        query_tpair, query_fact2tpair,
                        query_qpair, query_qual2qpair,
                        emb_ents, emb_rels, init_emb_ents, init_emb_rels
                    )
                    
                    if (loc + 1) % 2 == 0:
                        candidate, indices = torch.topk(ent_preds, k=top_k)
                    else:
                        candidate, indices = torch.topk(rel_preds, k=top_k)
                    
                    candidate_list = []
                    for num in indices.tolist()[0]:
                        if (loc + 1) % 2 == 0:
                            candidate_list.append(data_mape.get(train_graph.id2ent[num], train_graph.id2ent[num]))
                        else:
                            candidate_list.append(data_mapr.get(train_graph.id2rel[num], train_graph.id2rel[num]))
                    
                    facts_id_temp = facts_id.copy()
                    if loc + 1 < len(facts_id_temp):
                        facts_id_temp[loc + 1] = "[MASK]"
                        prompt4 = generate_prompt_question(facts_id_temp, candidate_list, dataset)
                        
                        response = call_openai_with_retry(
                            client, model_name, 
                            messages + [{"role": "user", "content": prompt4}],
                            max_retries=5, logger=logger
                        )
                        
                        try:
                            selected_idx = int(response.choices[0].message.content.strip())
                            if 0 <= selected_idx < len(indices.tolist()[0]):
                                replace = [indices.tolist()[0][selected_idx]]
                            else:
                                replace = [indices.tolist()[0][0]]
                        except ValueError:
                            replace = [indices.tolist()[0][0]]
                    else:
                        replace = [indices.tolist()[0][0]]

                    facts = query_pri.tolist()
                    for q, fact_idx in zip(query_qual.tolist(), query_qual2fact.tolist()):
                        if 0 <= fact_idx < len(facts):
                            facts[fact_idx].extend(q)
                
                fact = []
                for idx, j in enumerate(facts[0]):
                    if idx % 2 == 0:
                        fact.append(train_graph.id2ent[j])
                    else:
                        fact.append(train_graph.id2rel[j])
                
                batch_facts.append(fact)
                
                if len(batch_facts) >= save_interval:
                    save_progress(output_file, batch_facts, mode='a')
                    batch_facts = []
                
                time.sleep(1)
                
            except Exception as e:
                if batch_facts:
                    save_progress(output_file, batch_facts, mode='a')
                    batch_facts = []
                continue
        
        if batch_facts:
            save_progress(output_file, batch_facts, mode='a')
        
        return output_file


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
            output_file = f"./created_data/{dataset}/re-ranking_gpt_scratch_len{fact_len}.txt"
            result = generate_with_openai(
                log_name=log_name, 
                exp=exp, 
                dataset=dataset,
                data_dir=data_dir, 
                repeat_generate=repeat, 
                test_epoch=test_epoch,
                batch_num=1, 
                fact_len=fact_len,
                output_file=output_file,
                save_interval=1,
                model_name="gpt-5.2",
                dim=256,
                num_init_layer=4,
                num_head=16,
                num_layer=6,
                model_dropout=0.2
            )
        except Exception as e:
            continue
    
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
            output_file = f"./created_data/{dataset}/re-ranking_gpt_scratch_len{fact_len}.txt"
            result = generate_with_openai(
                log_name=log_name,
                exp=exp,
                dataset=dataset,
                data_dir=data_dir,
                repeat_generate=repeat,
                test_epoch=test_epoch,
                batch_num=1,
                fact_len=fact_len,
                output_file=output_file,
                save_interval=1,
                model_name="gpt-5.2",
                dim=256,
                num_init_layer=3,
                num_head=32,
                num_layer=4,
                model_dropout=0.1
            )
        except Exception as e:
            continue
    
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
            output_file = f"./created_data/{dataset}/re-ranking_gpt_scratch_len{fact_len}.txt"
            result = generate_with_openai(
                log_name=log_name,
                exp=exp,
                dataset=dataset,
                data_dir=data_dir,
                repeat_generate=repeat,
                test_epoch=test_epoch,
                batch_num=1,
                fact_len=fact_len,
                output_file=output_file,
                save_interval=1,
                model_name="gpt-5.2",
                dim=256,
                num_init_layer=3,
                num_head=32,
                num_layer=4,
                model_dropout=0.1
            )
            print(f"Completed: {result}")
        except Exception as e:
            continue


if __name__ == "__main__":
    main()