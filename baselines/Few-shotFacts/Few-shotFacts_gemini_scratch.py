from google import genai
from google.genai import types
import time
import os
import random
import sys
sys.path.append(os.path.dirname("../../"))
sys.path.append(os.path.dirname("../"))
from llm_utils import load_label_mappings, load_facts, convert_fact_to_natural_language, convert_fact_to_ids

os.environ["GEMINI_API_KEY"] = "API_KEY_HERE"  # Replace with your actual API key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "" # Replace with your actual credentials file path

def get_random_facts_containing_entity(seed_entity, facts, id_to_entity, id_to_relation, num_facts=30):
    containing_facts = []
    for fact in facts:
        if seed_entity in fact:
            containing_facts.append(fact)
    
    if len(containing_facts) < num_facts:
        selected_facts = containing_facts
    else:
        selected_facts = random.sample(containing_facts, num_facts)
    
    nl_facts = []
    for fact in selected_facts:
        nl_fact = convert_fact_to_natural_language(fact, id_to_entity, id_to_relation)
        nl_facts.append(nl_fact)
    
    return nl_facts


def generate_fact_with_context_facts(client, model, context_facts, fact_length):
    
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
    
    facts_str = "\n".join([f"[{', '.join(fact)}]" for fact in context_facts])
    
    prompt = f"""Role: You are an Expert Hyper-relational Fact Generator.

Here are {len(context_facts)} existing facts for context:
{facts_str}

Task: Generate a NEW hyper-relational fact with length {fact_length}

Constraint: Each fact must contain exactly {fact_length} elements. Only use use entities and relations that appear in the facts above.

Output format: Subject ; Relation ; Object ; Qualifier_relation ; Qualifier_entity ; ...
"""
    
    result = call_gemini(prompt)
    if not result:
        return None
    
    fact = [elem.strip() for elem in result.split(';')]
    
    return fact


def gemini_onehop_generator(num, fact_length, file_dir, output_file=None, model="gemini-3-pro-preview"):
    client = genai.Client()
    
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
            
            context_facts = get_random_facts_containing_entity(
                seed_entity_id, facts, id_to_entity, id_to_relation, num_facts=30
            )
            
            fact_nl = generate_fact_with_context_facts(
                client,
                model,
                context_facts,
                fact_length
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
        output_filename = f"Few-shotFacts_gemini_scratch_{dataset_name}.txt"
        output_dir = f"./created_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        print(f"Processing dataset: {dataset_name}")
        
        for fact_length, num_facts in fact_configs.items():
            gemini_onehop_generator(
                num=num_facts,
                fact_length=fact_length,
                file_dir=f"../../data/{dataset_name}",
                output_file=output_path, 
                model="gemini-3-pro-preview"
            )