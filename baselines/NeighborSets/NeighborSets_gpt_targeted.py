from openai import OpenAI
import time
import os
import random
from collections import defaultdict
import sys
sys.path.append(os.path.dirname("../../"))
sys.path.append(os.path.dirname("../"))
from llm_utils import load_label_mappings, load_facts, convert_fact_to_natural_language, convert_fact_to_ids
from mask_utils import load_masked_facts, load_train_facts, get_anchor_element_with_idx, is_entity_position
os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"  # Replace with your actual API key


def get_onehop_context(anchor_element, facts, id_to_entity, id_to_relation):
    entities = set()
    relations = set()
    
    for fact in facts:
        if anchor_element in fact:
            entities.add(id_to_entity.get(fact[0], fact[0]))
            entities.add(id_to_entity.get(fact[2], fact[2]))
            relations.add(id_to_relation.get(fact[1], fact[1]))
            
            if len(fact) > 3:
                for i in range(3, len(fact), 2):
                    if i < len(fact):
                        relations.add(id_to_relation.get(fact[i], fact[i]))
                    if i + 1 < len(fact):
                        entities.add(id_to_entity.get(fact[i + 1], fact[i + 1]))
    
    return list(entities), list(relations)


def generate_fact_batch_with_onehop(client, model, masked_fact, anchor_element, anchor_idx, entities, relations, id_to_entity, id_to_relation, num_candidates=10):
    
    def call_openai(prompt_text):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                temperature=1.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return None
    
    if is_entity_position(anchor_idx):
        anchor_nl = id_to_entity.get(anchor_element, anchor_element)
        element_type = "entity"
    else:
        anchor_nl = id_to_relation.get(anchor_element, anchor_element)
        element_type = "relation"
    
    entities_str = ", ".join(entities)
    relations_str = ", ".join(relations)
    
    masked_fact_nl = []
    for idx, elem in enumerate(masked_fact):
        if elem == 'MASK':
            masked_fact_nl.append('MASK')
        elif is_entity_position(idx):
            masked_fact_nl.append(id_to_entity.get(elem, elem))
        else:
            masked_fact_nl.append(id_to_relation.get(elem, elem))
    
    masked_fact_str = ' ; '.join(masked_fact_nl)
    fact_length = len(masked_fact)
    
    prompt = f"""Role: You are an Expert Hyper-relational Fact Generator.

Available entities: [{entities_str}]
Available relations: [{relations_str}]

Task: Generate {num_candidates} DIFFERENT hyper-relational facts by completing the following MASKED fact. Each fact should be unique.

Masked Fact: {masked_fact_str}

Constraint: 
- Each fact must contain exactly {fact_length} elements and have the same number of elements as the masked fact
- Position {anchor_idx} MUST be "{anchor_nl}" (this is fixed and cannot be changed)
- Only use entities and relations that appear in the context facts above
- Generate {num_candidates} DIFFERENT facts

Output format: Provide exactly {num_candidates} facts, one per line:
Subject ; Relation ; Object ; Qualifier_relation ; Qualifier_entity ; ...
Subject ; Relation ; Object ; Qualifier_relation ; Qualifier_entity ; ...
...

Provide ONLY {num_candidates} completed facts, no explanations.
"""
    
    result = call_openai(prompt)
    if not result:
        return []
    
    candidate_facts = []
    lines = result.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and ';' in line:
            fact = [elem.strip() for elem in line.split(';')]
            candidate_facts.append(fact)
    
    return candidate_facts


def openai_onehop_generator(masked_file, file_dir, output_file=None, model="gpt-5.2", num_candidates=10):
    client = OpenAI()
    
    id_to_entity, id_to_relation, entity_to_id, relation_to_id = load_label_mappings(file_dir)
    
    facts = load_facts(file_dir)
    
    train_facts = load_train_facts(file_dir)

    masked_facts = load_masked_facts(masked_file)
    
    file_handle = None
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        file_handle = open(output_file, 'a', encoding='utf-8')
    
    generated_facts = []
    
    try:
        for i, masked_fact in enumerate(masked_facts):

            anchor_element, anchor_idx = get_anchor_element_with_idx(masked_fact)
            
            if anchor_element is None:
                continue
            
            onehop_entities, onehop_relations = get_onehop_context(
                anchor_element, facts, id_to_entity, id_to_relation
            )
            
            if not onehop_entities or not onehop_relations:
                continue
            
            candidate_facts_nl = generate_fact_batch_with_onehop(
                client,
                model,
                masked_fact,
                anchor_element,
                anchor_idx,
                onehop_entities,
                onehop_relations,
                id_to_entity,
                id_to_relation,
                num_candidates=num_candidates
            )
            
            if not candidate_facts_nl:
                fact_ids = masked_fact
            else:
                
                all_in_training = True
                for fact_nl in candidate_facts_nl:
                    try:
                        candidate_ids = convert_fact_to_ids(fact_nl, entity_to_id, relation_to_id)
                        if tuple(candidate_ids) not in train_facts:
                            all_in_training = False
                            break
                    except:
                        all_in_training = False
                        break
                
                if all_in_training:
                    fact_ids = masked_fact
                else:
                    fact_ids = None
                    has_valid_candidate = False
                    
                    for fact_nl in candidate_facts_nl:
                        try:
                            candidate_ids = convert_fact_to_ids(fact_nl, entity_to_id, relation_to_id)
                            
                            if tuple(candidate_ids) in train_facts:
                                continue
                            else:
                                fact_ids = candidate_ids
                                has_valid_candidate = True
                                break
                        except Exception as e:
                            continue
                    
                    if not has_valid_candidate:
                        fact_ids = ["PARSING_ERROR"]
            
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
    datasets = ["wd50k-eval", "WikiPeople", "WikiPeople--eval"]
    
    for dataset_name in datasets:
        masked_file = f"../../data/{dataset_name}/targeted.txt"
        output_filename = f"NeighborSets_gpt_targeted_{dataset_name}.txt"
        output_dir = f"./created_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        print(f"Processing dataset: {dataset_name}")        
        openai_onehop_generator(
            masked_file=masked_file,
            file_dir=f"../../data/{dataset_name}",
            output_file=output_path, 
            model="gpt-5.2",
            num_candidates=10
        )
        
        print(f"Completed: {dataset_name}")