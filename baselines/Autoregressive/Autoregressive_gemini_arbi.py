from google import genai
from google.genai import types
import time
import os
import random
from collections import defaultdict
import sys
sys.path.append(os.path.dirname("../../"))
sys.path.append(os.path.dirname("../"))
from llm_utils import load_label_mappings, load_facts, convert_fact_to_natural_language, convert_fact_to_ids
from mask_utils import load_masked_facts, load_train_facts, is_entity_position, get_anchor_element, has_mask

os.environ["GEMINI_API_KEY"] = "API_KEY_HERE"  # Replace with your actual API key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "" # Replace with your actual credentials file path

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


def format_facts_as_examples(facts, id_to_entity, id_to_relation, max_examples=30):
    
    sampled_facts = random.sample(facts, min(len(facts), max_examples))
    
    nl_facts = []
    for fact in sampled_facts:
        nl_fact = convert_fact_to_natural_language(fact, id_to_entity, id_to_relation)
        nl_facts.append(str(nl_fact))
    return "\n".join(nl_facts)

def generate_sequential_fact_from_masked(client, model, masked_fact, entity_index, relation_index, 
                                        id_to_entity, id_to_relation, entity_to_id, relation_to_id):
    anchor_elements = get_anchor_element(masked_fact)
    
    if not anchor_elements:
        return None
    
    anchor_indices = set()
    if anchor_elements:
        anchor_indices = {idx for _, idx in anchor_elements}
    
    current_fact_ids = list(masked_fact)
    current_fact_nl = []
    for idx, elem in enumerate(masked_fact):
        if elem == 'MASK':
            current_fact_nl.append('MASK')
        elif is_entity_position(idx):
            current_fact_nl.append(id_to_entity.get(elem, elem))
        else:
            current_fact_nl.append(id_to_relation.get(elem, elem))
    
    conversation = []
    
    anchor_facts = []
    for anchor_element, anchor_idx in anchor_elements:
        if is_entity_position(anchor_idx):
            anchor_facts.extend(entity_index.get(anchor_element, []))
        else:
            anchor_facts.extend(relation_index.get(anchor_element, []))
    
    anchor_facts = list(set(tuple(f) for f in anchor_facts))
    anchor_facts = [list(f) for f in anchor_facts]
    
    if not anchor_facts:
        return None
    
    anchor_examples = format_facts_as_examples(anchor_facts, id_to_entity, id_to_relation, 
                                              max_examples=30)
    
    anchor_descriptions = []
    for anchor_element, anchor_idx in anchor_elements:
        if is_entity_position(anchor_idx):
            anchor_nl = id_to_entity.get(anchor_element, anchor_element)
            element_type = "entity"
        else:
            anchor_nl = id_to_relation.get(anchor_element, anchor_element)
            element_type = "relation"
        anchor_descriptions.append(f"Position {anchor_idx}: {element_type} '{anchor_nl}'")
    
    anchor_info = ", ".join(anchor_descriptions)
    
    for fill_idx in range(len(masked_fact)):
        if masked_fact[fill_idx] != 'MASK':
            continue
        
        is_entity = is_entity_position(fill_idx)
        element_type = "entity" if is_entity else "relation"
        
        last_filled_idx = fill_idx - 1
        while last_filled_idx >= 0 and (current_fact_ids[last_filled_idx] == 'MASK' or last_filled_idx in anchor_indices):
            last_filled_idx -= 1
        
        if last_filled_idx >= 0:
            last_element_id = current_fact_ids[last_filled_idx]
            
            if is_entity_position(last_filled_idx):
                last_facts = entity_index.get(last_element_id, [])
            else:
                last_facts = relation_index.get(last_element_id, [])
            
            if not last_facts:
                last_facts = anchor_facts
            
            context_examples = format_facts_as_examples(last_facts, id_to_entity, id_to_relation, 
                                                       max_examples=30)
        else:
            context_examples = anchor_examples
        
        current_state = []
        for idx, elem in enumerate(current_fact_nl):
            if idx < fill_idx:
                current_state.append(elem)
            elif idx == fill_idx:
                current_state.append(f"[FILLING NOW]")
            else:
                current_state.append("MASK")
        current_state_str = " ; ".join(current_state)
        
        prompt = f"""
Role: You are an Expert Hyper-relational Fact Generator.

Current fact being built: {current_state_str}
Fixed positions (CANNOT change): {anchor_info}

Here are facts for context:
{context_examples}

Task: Select ONE {element_type} from the facts above that could fill the [FILLING NOW] position.
You must only use {element_type}s that appear in the provided facts and the facts shown in previous steps.

Do not create same facts as in the examples.

Output ONLY the {element_type} name, nothing else.
"""
        
        conversation.append({"role": "user", "parts": [{"text": prompt}]})
        
        response = client.models.generate_content(
            model=model,
            contents=conversation,
            config=types.GenerateContentConfig(
                temperature=1.0,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW
                )
            )
        )
        
        filled_element_nl = response.text.strip()
        
        try:
            if is_entity:
                filled_element_id = entity_to_id.get(filled_element_nl, filled_element_nl)
            else:
                filled_element_id = relation_to_id.get(filled_element_nl, filled_element_nl)
        except:
            filled_element_id = filled_element_nl

        current_fact_ids[fill_idx] = filled_element_id
        current_fact_nl[fill_idx] = filled_element_nl
        
        conversation.append({"role": "model", "parts": [{"text": filled_element_nl}]})
    
    return current_fact_ids


def gemini_sequential(masked_file, file_dir, output_file=None, model="gemini-3-pro-preview", max_retries=3):
    client = genai.Client()
    
    id_to_entity, id_to_relation, entity_to_id, relation_to_id = load_label_mappings(file_dir)
    
    facts = load_facts(file_dir)
    entity_index, relation_index = build_indices(facts)
    
    train_facts = load_train_facts(file_dir)

    masked_facts = load_masked_facts(masked_file)
    
    file_handle = None
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        file_handle = open(output_file, 'a', encoding='utf-8')
    
    generated_facts = []
    
    try:
        for i, masked_fact in enumerate(masked_facts):
            if not has_mask(masked_fact):
                generated_facts.append(masked_fact)
                fact_str = str(masked_fact)
                
                if file_handle:
                    file_handle.write(fact_str + "\n")
                    file_handle.flush()
                
                continue
            
            fact_ids = None
            previously_generated = []
            
            for attempt in range(max_retries):
                try:
                    fact_ids = generate_sequential_fact_from_masked(
                        client, model, masked_fact, entity_index, relation_index,
                        id_to_entity, id_to_relation, entity_to_id, relation_to_id
                    )
                    
                    if fact_ids is None:
                        continue
                    
                    previously_generated.append(fact_ids)
                    
                    has_unconverted = any(
                        (is_entity_position(idx) and elem not in entity_to_id.values()) or
                        (not is_entity_position(idx) and elem not in relation_to_id.values())
                        for idx, elem in enumerate(fact_ids) if elem != 'MASK'
                    )
                    
                    if has_unconverted:
                        break
                    
                    if tuple(fact_ids) not in train_facts:
                        break
                    else:
                        fact_ids = None
                
                except Exception as e:
                    print(f"Error during generation: {e}")
                    fact_ids = None
                    continue
            
            if fact_ids is None:
                fact_ids = masked_fact
            
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
    datasets = ["wd50k-eval","WikiPeople","WikiPeople--eval"]
    
    for dataset_name in datasets:
        masked_file = f"../../data/{dataset_name}/arbitrary_masking.txt"
        output_filename = f"Autoregressive_gemini_arbi_{dataset_name}.txt"
        output_dir = f"./created_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        print(f"Processing dataset: {dataset_name}")
        
        gemini_sequential(
            masked_file=masked_file,
            file_dir=f"../../data/{dataset_name}",
            output_file=output_path,
            model="gemini-3-pro-preview",
            max_retries=10
        )
        
        print(f"Completed: {dataset_name}")