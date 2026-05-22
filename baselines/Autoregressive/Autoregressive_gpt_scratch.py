from openai import OpenAI
import time
import os
import random
from collections import defaultdict
import sys
sys.path.append(os.path.dirname("../../"))
sys.path.append(os.path.dirname("../"))
from llm_utils import load_label_mappings, load_facts, convert_fact_to_natural_language, convert_fact_to_ids    

os.environ['OPENAI_API_KEY'] = "API_KEY_HERE"  # Replace with your actual API key

def build_indices(facts):
    entity_index = defaultdict(list)
    relation_index = defaultdict(list)
    
    for fact in facts:
        h, r, t = fact[0], fact[1], fact[2]
        entity_index[h].append(fact)
        entity_index[t].append(fact)
        relation_index[r].append(fact)
        
        if len(fact) > 3:
            for i in range(3, len(fact), 2):
                q_rel = fact[i]
                q_val = fact[i + 1]
                relation_index[q_rel].append(fact)
                entity_index[q_val].append(fact)
    
    return entity_index, relation_index


def format_facts_as_examples(facts, id_to_entity, id_to_relation, max_examples=None):
    if max_examples is None:
        max_examples = len(facts)
    sampled_facts = random.sample(facts, min(len(facts), max_examples))
    
    nl_facts = []
    for fact in sampled_facts:
        nl_fact = convert_fact_to_natural_language(fact, id_to_entity, id_to_relation)
        nl_facts.append(str(nl_fact))
    return "\n".join(nl_facts)

def generate_sequential_fact(client, model, entity_index, relation_index, 
                            fact_length, 
                            id_to_entity, id_to_relation, entity_to_id, relation_to_id):
    
    all_entities = list(entity_index.keys())
    head_id = random.choice(all_entities)
    head_nl = id_to_entity.get(head_id, head_id)
    
    messages = []
    
    head_facts = entity_index.get(head_id, [])
    head_examples = format_facts_as_examples(head_facts, id_to_entity, id_to_relation, max_examples=30)
    
    prompt_relation = f"""
Role: You are an Expert Hyper-relational Fact Generator.

The HEAD entity has been selected: {head_nl}

Here are facts where "{head_nl}" appears:
{head_examples}

Task: Select ONE relation from the facts above that could connect to this head entity.
You must only use relations that appear in the provided facts.

Output ONLY the relation name, nothing else.
"""
    
    messages.append({"role": "user", "content": prompt_relation})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    relation_nl = response.choices[0].message.content.strip()
    relation_id = relation_to_id.get(relation_nl, relation_nl)
    
    messages.append({"role": "assistant", "content": relation_nl})

    current_fact_nl = [head_nl, relation_nl]
    
    relation_facts = relation_index.get(relation_id, [])
    relation_examples = format_facts_as_examples(relation_facts, id_to_entity, id_to_relation, max_examples=30)
    
    current_fact_str = str(current_fact_nl)
    
    prompt_tail = f"""
Current fact being built: {current_fact_str}

Here are facts where "{relation_nl}" appears:
{relation_examples}

Task: Select ONE entity from the facts above that could connect to this relation.
You must only use entities that appear in the provided facts and the facts shown in previous steps.

Output ONLY the entity name, nothing else.
"""
    
    messages.append({"role": "user", "content": prompt_tail})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    tail_nl = response.choices[0].message.content.strip()
    tail_id = entity_to_id.get(tail_nl, tail_nl)
    
    messages.append({"role": "assistant", "content": tail_nl})
    
    current_fact_nl = [head_nl, relation_nl, tail_nl]
    num_qualifiers = (fact_length - 3) // 2
    last_element_nl = tail_nl
    last_element_id = tail_id
    
    for i in range(num_qualifiers):
        last_facts = entity_index.get(last_element_id, []) if last_element_id in entity_index else relation_index.get(last_element_id, [])
        last_examples = format_facts_as_examples(last_facts, id_to_entity, id_to_relation, max_examples=30)
        
        current_fact_str = str(current_fact_nl)
        
        prompt_q_rel = f"""
Current fact being built: {current_fact_str}

Here are facts where "{last_element_nl}" appears:
{last_examples}

Task: Select ONE relation from the facts above that could connect to this entity.
You must only use relations that appear in the provided facts and the facts shown in previous steps.

Output ONLY the relation name, nothing else.
"""
        
        messages.append({"role": "user", "content": prompt_q_rel})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        q_relation_nl = response.choices[0].message.content.strip()
        q_relation_id = relation_to_id.get(q_relation_nl, q_relation_nl)
        current_fact_nl.append(q_relation_nl)
        
        messages.append({"role": "assistant", "content": q_relation_nl})
        
        q_rel_facts = relation_index.get(q_relation_id, [])
        q_rel_examples = format_facts_as_examples(q_rel_facts, id_to_entity, id_to_relation, max_examples=30)
        
        current_fact_str = str(current_fact_nl)
        
        prompt_q_val = f"""
Current fact being built: {current_fact_str}

Here are facts where "{q_relation_nl}" appears:
{q_rel_examples}

Task: Select ONE entity from the facts above that could connect to this relation.
You must only use entities that appear in the provided facts and the facts shown in previous steps.

Output ONLY the entity name, nothing else.
"""
        
        messages.append({"role": "user", "content": prompt_q_val})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        q_value_nl = response.choices[0].message.content.strip()
        q_value_id = entity_to_id.get(q_value_nl, q_value_nl)
        current_fact_nl.append(q_value_nl)
        
        messages.append({"role": "assistant", "content": q_value_nl})
        
        last_element_nl = q_value_nl
        last_element_id = q_value_id
    
    return current_fact_nl


def openai_sequential(num, fact_length, file_dir, output_file=None, model="gpt-5.2"):
    client = OpenAI()
    
    id_to_entity, id_to_relation, entity_to_id, relation_to_id = load_label_mappings(file_dir)
    
    
    facts = load_facts(file_dir)
    entity_index, relation_index = build_indices(facts)
    
    file_handle = None
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        file_handle = open(output_file, 'a', encoding='utf-8')
    
    generated_facts = []
    
    try:
        for i in range(num):
            
            fact_nl = generate_sequential_fact(
                client, model, entity_index, relation_index, 
                fact_length,
                id_to_entity, id_to_relation, entity_to_id, relation_to_id
            )
            
            fact_ids = convert_fact_to_ids(fact_nl, entity_to_id, relation_to_id)
            
            generated_facts.append(fact_ids)
            fact_str = str(fact_ids)

            
            if file_handle:
                file_handle.write(fact_str + "\n")
                file_handle.flush()
    
    finally:
        if file_handle:
            file_handle.close()
    
    return generated_facts


if __name__ == "__main__":
    datasets = {
        "wd50k-eval": {
            3: 864,
            5: 89,
            7: 40,
            9: 4,
            11: 2,
            13: 1
        },
        "WikiPeople": {
            3: 884,
            5: 67,
            7: 40,
            9: 7,
            11: 2
        },
        "WikiPeople--eval": {
            3: 974,
            5: 20,
            7: 5,
            9: 1
        }
    }
    
    for dataset_name, fact_configs in datasets.items():
        output_filename = f"Autoregressive_gpt_scratch_{dataset_name}.txt"
        output_dir = f"./created_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing dataset: {dataset_name}")
        
        for fact_length, num_facts in fact_configs.items():
            
            generated_facts = openai_sequential(
                num=num_facts,
                fact_length=fact_length,
                file_dir=f"../../data/{dataset_name}",
                output_file=output_path,
                model="gpt-5.2"
            )