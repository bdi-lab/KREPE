from openai import OpenAI
import os
import random
import sys
sys.path.append(os.path.dirname("../../"))
sys.path.append(os.path.dirname("../"))
from llm_utils import load_label_mappings, convert_fact_to_natural_language, convert_fact_to_ids
from mask_utils import load_masked_facts, load_train_facts, is_entity_position

os.environ['OPENAI_API_KEY'] = 'API_KEY_HERE'  # Replace with your actual API key


def openai_naive(masked_file, train_file, file_dir, output_path, model="gpt-5.2", batch_size=100, num_candidates=10):
    client = OpenAI()
    
    id_to_entity, id_to_relation, entity_to_id, relation_to_id = load_label_mappings(file_dir)
    
    train_facts = load_train_facts(file_dir)
    
    masked_facts = load_masked_facts(masked_file)
    
    train_file_path = os.path.join(file_dir, train_file)
    with open(train_file_path, 'r', encoding='utf-8') as f:
        all_facts = f.readlines()
    
    remaining_masked = len(masked_facts)
    masked_idx = 0
    
    while remaining_masked > 0:
        current_batch_size = min(batch_size, remaining_masked)
        current_masked_batch = masked_facts[masked_idx:masked_idx + current_batch_size]
        
        sampled_facts = random.sample(all_facts, min(1000, len(all_facts)))
        
        nl_facts = []
        for fact_line in sampled_facts:
            fact_elements = fact_line.strip().split('\t')
            if len(fact_elements) >= 3:
                nl_fact = convert_fact_to_natural_language(fact_elements, id_to_entity, id_to_relation)
                nl_facts.append(nl_fact)
        
        facts_str = '\n'.join([', '.join(nf) for nf in nl_facts])
        
        masked_facts_str = []
        for masked_fact in current_masked_batch:
            masked_fact_nl = []
            for idx, elem in enumerate(masked_fact):
                if elem == 'MASK':
                    masked_fact_nl.append('MASK')
                elif is_entity_position(idx):
                    masked_fact_nl.append(id_to_entity.get(elem, elem))
                else:
                    masked_fact_nl.append(id_to_relation.get(elem, elem))
            masked_facts_str.append(' ; '.join(masked_fact_nl))
        
        masked_list = '\n'.join([mf for mf in masked_facts_str])
        
        prompt_text = f"""Role: You are an Expert Hyper-relational Fact Generator.
Here are 1000 hyper-relational facts for context:
{facts_str}

Task: Complete the following {current_batch_size} MASKED facts. For EACH masked fact, generate {num_candidates} DIFFERENT hyper-relational facts by completing the following MASKED fact. Each fact should be unique.

Masked facts to complete:
{masked_list}

Constraint:
- For each masked fact, maintain its exact length
- Keep non-MASK elements in their original positions
- Only use entities and relations from the context facts above
- Generate {num_candidates} different completions for EACH masked fact

Output format: For each masked fact, provide {num_candidates} completions, one per line:
Fact 1 - Completion 1: Subject ; Relation ; Object ; ...
Fact 1 - Completion 2: Subject ; Relation ; Object ; ...
...
Fact 1 - Completion {num_candidates}: Subject ; Relation ; Object ; ...
Fact 2 - Completion 1: Subject ; Relation ; Object ; ...
...

Provide ONLY completed facts in the format above, no explanations.
"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            temperature=1.0,
        )
        
        response_text = response.choices[0].message.content
        lines = response_text.strip().split('\n')
        
        candidates_per_masked = {}
        current_fact_idx = 0
        
        for line in lines:
            line = line.strip()
            if not line or ';' not in line:
                continue
            
            if ':' in line:
                line = line.split(':', 1)[1].strip()
            
            fact_elements = [elem.strip() for elem in line.split(';')]
            
            if current_fact_idx not in candidates_per_masked:
                candidates_per_masked[current_fact_idx] = []
            
            candidates_per_masked[current_fact_idx].append(fact_elements)
            
            if len(candidates_per_masked[current_fact_idx]) >= num_candidates:
                current_fact_idx += 1
                if current_fact_idx >= current_batch_size:
                    break
        
        file_handle = open(output_path, 'a', encoding='utf-8')
        
        for i in range(current_batch_size):
            masked_fact = current_masked_batch[i]
            
            if i not in candidates_per_masked or not candidates_per_masked[i]:
                fact_str = str(masked_fact)
                file_handle.write(fact_str + '\n')
                file_handle.flush()
                continue
            
            all_in_training = True
            for candidate_nl in candidates_per_masked[i]:
                try:
                    candidate_ids = convert_fact_to_ids(candidate_nl, entity_to_id, relation_to_id)
                    if tuple(candidate_ids) not in train_facts:
                        all_in_training = False
                        break
                except:
                    all_in_training = False
                    break
            
            if all_in_training:
                fact_str = str(masked_fact)
                file_handle.write(fact_str + '\n')
                file_handle.flush()
            else:
                found_unique = False
                
                for candidate_nl in candidates_per_masked[i]:
                    try:
                        candidate_ids = convert_fact_to_ids(candidate_nl, entity_to_id, relation_to_id)
                        
                        if tuple(candidate_ids) in train_facts:
                            continue
                        else:
                            fact_str = str(candidate_ids)
                            file_handle.write(fact_str + '\n')
                            file_handle.flush()
                            found_unique = True
                            break
                    except Exception as e:
                        continue
                
                if not found_unique:
                    fact_str = str(["PARSING_ERROR"])
                    file_handle.write(fact_str + '\n')
                    file_handle.flush()
        
        file_handle.close()
        
        masked_idx += current_batch_size
        remaining_masked -= current_batch_size


if __name__ == "__main__":
    datasets = ["wd50k-eval", "WikiPeople", "WikiPeople--eval"]
    
    for dataset_name in datasets:
        
        masked_file = f"../../data/{dataset_name}/targeted.txt"
        output_filename = f"RandomFacts_gpt_targeted_{dataset_name}.txt"
        output_dir = f"./created_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        print(f"Processing dataset: {dataset_name}")        
        openai_naive(
            masked_file=masked_file,
            train_file="train.txt",
            file_dir=f"../../data/{dataset_name}",
            output_path=output_path,
            model="gpt-5.2",
            batch_size=100,
            num_candidates=10
        )
        
        print(f"Completed: {dataset_name}")