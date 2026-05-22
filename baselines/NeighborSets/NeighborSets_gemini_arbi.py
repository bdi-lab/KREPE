from google import genai
from google.genai import types
import time
import os
import sys
sys.path.append(os.path.dirname("../../"))
sys.path.append(os.path.dirname("../"))
from llm_utils import load_label_mappings, load_facts, convert_fact_to_natural_language, convert_fact_to_ids
from mask_utils import load_masked_facts, load_train_facts, get_anchor_element, is_entity_position
os.environ["GEMINI_API_KEY"] = "API_KEY_HERE"  # Replace with your actual API key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "" # Replace with your actual credentials file path


def get_onehop_context(anchor_elements, facts, id_to_entity, id_to_relation):
    entities = set()
    relations = set()
    
    for fact in facts:
        fact_contains_anchor = False
        for anchor_element, _ in anchor_elements:
            if anchor_element in fact:
                fact_contains_anchor = True
                break
        
        if fact_contains_anchor:
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


def generate_fact_batch_with_onehop(client, model, masked_fact, anchor_elements, entities, relations, id_to_entity, id_to_relation, num_candidates=10):
    
    def call_gemini(prompt_text):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    temperature=1.0,
                        thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                    )
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"GenAI API Error: {e}")
            return None
    
    anchor_descriptions = []
    for anchor_element, anchor_idx in anchor_elements:
        if is_entity_position(anchor_idx):
            anchor_nl = id_to_entity.get(anchor_element, anchor_element)
            element_type = "entity"
        else:
            anchor_nl = id_to_relation.get(anchor_element, anchor_element)
            element_type = "relation"
        anchor_descriptions.append(f"Position {anchor_idx}: {element_type} '{anchor_nl}'")
    
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
    
    anchor_info = "\n".join([f"- {desc}" for desc in anchor_descriptions])
    
    prompt = f"""Role: You are an Expert Hyper-relational Fact Generator.

Available entities: [{entities_str}]
Available relations: [{relations_str}]

Task: Generate {num_candidates} DIFFERENT hyper-relational facts by completing the following MASKED fact. Each fact should be unique.

Masked Fact: {masked_fact_str}

Constraints: 
- Each fact must contain exactly {fact_length} elements and have the same number of elements as the masked fact
- The following positions are FIXED and CANNOT be changed:
{anchor_info}
- Only use entities and relations that appear in the available entities and relations above
- Generate {num_candidates} DIFFERENT facts

Output format: Provide exactly {num_candidates} facts, one per line:
Subject ; Relation ; Object ; Qualifier_relation ; Qualifier_entity ; ...
Subject ; Relation ; Object ; Qualifier_relation ; Qualifier_entity ; ...
...

Provide ONLY {num_candidates} completed facts, no explanations.
"""
    
    result = call_gemini(prompt)
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


def gemini_onehop_generator(masked_file, file_dir, output_file=None, model="gemini-3-pro-preview", num_candidates=10):
    client = genai.Client()
    
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
            
            anchor_elements = get_anchor_element(masked_fact)
            
            if not anchor_elements:
                continue
            
            
            onehop_entities, onehop_relations = get_onehop_context(
                anchor_elements, facts, id_to_entity, id_to_relation
            )
            
            if not onehop_entities or not onehop_relations:
                continue
            
            candidate_facts_nl = generate_fact_batch_with_onehop(
                client,
                model,
                masked_fact,
                anchor_elements,
                onehop_entities,
                onehop_relations,
                id_to_entity,
                id_to_relation,
                num_candidates=num_candidates
            )
            
            if not candidate_facts_nl:
                print(f"Failed to generate candidate facts")
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
                    
                    for idx, fact_nl in enumerate(candidate_facts_nl):
                        try:
                            candidate_ids = convert_fact_to_ids(fact_nl, entity_to_id, relation_to_id)
                            
                            if tuple(candidate_ids) in train_facts:
                                continue
                            else:
                                fact_ids = candidate_ids
                                has_valid_candidate = True
                                break
                        except Exception as e:
                            print(f"Error during generation: {e}")
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
    datasets = ["wd50k-eval","WikiPeople","WikiPeople--eval"]
    
    for dataset_name in datasets:
        masked_file = f"../../data/{dataset_name}/arbitrary_masking.txt"
        output_filename = f"NeighborSets_gemini_arbi_{dataset_name}.txt"
        output_dir = f"./created_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        print(f"Processing dataset: {dataset_name}")        
        gemini_onehop_generator(
            masked_file=masked_file,
            file_dir=f"../../data/{dataset_name}",
            output_file=output_path, 
            model="gemini-3-pro-preview",
            num_candidates=10
        )
        
        print(f"Completed: {dataset_name}")