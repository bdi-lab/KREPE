from openai import OpenAI
import os
import random
import sys
sys.path.append(os.path.dirname("../../"))
sys.path.append(os.path.dirname("../"))
from llm_utils import load_label_mappings, convert_fact_to_natural_language, convert_fact_to_ids

os.environ['OPENAI_API_KEY'] = 'API_KEY_HERE'  # Replace with your actual API key

def openai_naive(num, fact_length, train_file, file_dir, output_path, model="gpt-4"):
    client = OpenAI()
    
    id_to_entity, id_to_relation, entity_to_id, relation_to_id = load_label_mappings(file_dir)
    
    train_file_path = os.path.join(file_dir, train_file)
    
    with open(train_file_path, 'r', encoding='utf-8') as f:
        all_facts = f.readlines()
    
    remaining = num
    
    while remaining > 0:
        batch_size = min(100, remaining)
        sampled_facts = random.sample(all_facts, min(1000, len(all_facts)))
        
        nl_facts = []
        for fact_line in sampled_facts:
            fact_elements = fact_line.strip().split()
            if len(fact_elements) >= 3:
                nl_fact = convert_fact_to_natural_language(fact_elements, id_to_entity, id_to_relation)
                nl_facts.append(nl_fact)
        
        facts_str = '\n'.join([', '.join(nf) for nf in nl_facts])
        
        prompt_text = f"""Role: You are an Expert Hyper-relational Fact Generator.
        
Here are 1000 hyper-relational facts for context:
{facts_str}

Task: Generate {batch_size} NEW facts with target length {fact_length} based on this list.
Use entities, relations, and qualifier key-values only from the provided list.

Constraint: Each fact must contain exactly {fact_length} elements.

Output format: Subject ; Relation ; Object ; Qualifier_relation ; Qualifier_entity ; ...
"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt_text}
            ]
        )
        
        response_text = response.choices[0].message.content
        lines = response_text.strip().split('\n')
        
        converted_facts = []
        for line in lines:
            line = line.strip()
            
            fact_elements = [elem.strip() for elem in line.split(';')]
            
            id_fact = convert_fact_to_ids(fact_elements, entity_to_id, relation_to_id)
            fact_str = ', '.join(id_fact)
            fact_str = '[' + fact_str + ']'
            converted_facts.append(fact_str)
        
        with open(output_path, 'a', encoding='utf-8') as f:
            for fact in converted_facts:
                f.write(fact + '\n')
        
        remaining -= batch_size


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
        print(f"Processing dataset: {dataset_name}")
        
        output_filename = f"RandomFacts_gpt_scratch_{dataset_name}.txt"
        output_dir = f"./created_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)   
        
        for fact_length, num_facts in fact_configs.items():
            openai_naive(
                num=num_facts,
                fact_length=fact_length,
                train_file="train.txt",
                file_dir=f"../../data/{dataset_name}",
                output_path=output_path,
                model="gpt-5.2"
            )