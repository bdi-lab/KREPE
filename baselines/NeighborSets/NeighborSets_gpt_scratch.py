from openai import OpenAI
import time
import os
import random
from collections import defaultdict
import sys
sys.path.append(os.path.dirname("../../"))
sys.path.append(os.path.dirname("../"))
from llm_utils import load_label_mappings, load_facts, convert_fact_to_natural_language, convert_fact_to_ids
os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"  # Replace with your actual API key

def get_onehop_context(seed_entity, facts, id_to_entity, id_to_relation):
    entities = set()
    relations = set()
    
    for fact in facts:
        if seed_entity in fact:
            entities.add(id_to_entity.get(fact[0], fact[0]))
            entities.add(id_to_entity.get(fact[2], fact[2]))
            
            relations.add(id_to_relation.get(fact[1], fact[1]))
            
            if len(fact) > 3:
                for i in range(3, len(fact), 2):
                    if i < len(fact):
                        relations.add(id_to_relation.get(fact[i], fact[i]))
                    if i + 1 < len(fact):
                        entities.add(id_to_entity.get(fact[i + 1], fact[i + 1]))
    
    seed_entity_nl = id_to_entity.get(seed_entity, seed_entity)
    entities.discard(seed_entity_nl)
    
    return list(entities), list(relations)


def generate_fact_with_onehop(client, model, seed_entity_nl, entities, relations, fact_length):
    
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
    
    entities_str = ", ".join(entities)
    relations_str = ", ".join(relations)
    
    prompt = f"""Role: You are an Expert Hyper-relational Fact Generator.

Task: Generate a hyper-relational fact with length {fact_length} containing the [{entities_str}] and [{relations_str}]

Constraint: Each fact must contain exactly {fact_length} elements. Only use the given entities and relations.

Output format: Subject ; Relation ; Object ; Qualifier_relation ; Qualifier_entity ; ...

"""
    
    result = call_openai(prompt)
    if not result:
        return None
    
    fact = [elem.strip() for elem in result.split(';')]
    
    return fact


def openai_onehop_generator(num, fact_length, file_dir, output_file=None, model="gpt-4"):
    client = OpenAI()
    
    id_to_entity, id_to_relation, entity_to_id, relation_to_id = load_label_mappings(file_dir)
    
    facts = load_facts(file_dir)
    
    all_entities = set()
    for fact in facts:
        all_entities.add(fact[0])
        all_entities.add(fact[2])
        if len(fact) > 3:
            for i in range(4, len(fact), 2):
                all_entities.add(fact[i])
    all_entities = list(all_entities)
    
    file_handle = None
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        file_handle = open(output_file, 'a', encoding='utf-8')
    
    generated_facts = []
    
    try:
        for i in range(num):
            seed_entity_id = random.choice(all_entities)
            seed_entity_nl = id_to_entity.get(seed_entity_id, seed_entity_id)
            
            onehop_entities, onehop_relations = get_onehop_context(
                seed_entity_id, facts, id_to_entity, id_to_relation
            )
            
            fact_nl = generate_fact_with_onehop(
                client,
                model,
                seed_entity_nl,
                onehop_entities,
                onehop_relations,
                fact_length
            )
            
            fact_ids = convert_fact_to_ids(fact_nl, entity_to_id, relation_to_id)
            
            generated_facts.append(fact_ids)
            fact_str = str(fact_ids)
            
            if file_handle:
                file_handle.write(fact_str + "\n")
                file_handle.flush()
            
            time.sleep(0.5)
            
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
        output_filename = f"NeighborSets_gpt_scratch_{dataset_name}.txt"
        output_dir = f"./created_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        print(f"Processing dataset: {dataset_name}")        
        
        for fact_length, num_facts in fact_configs.items():
            generated_facts = openai_onehop_generator(
                num=num_facts,
                fact_length=fact_length,
                file_dir=f"../../data/{dataset_name}",
                output_file=output_path, 
                model="gpt-5.2"
            )