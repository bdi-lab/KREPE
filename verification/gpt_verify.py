from openai import OpenAI
import os
import json
import ast
from verification_prompt import generate_prompt
from datetime import datetime
os.environ['OPENAI_API_KEY'] = "API_KEY_HERE"  # Replace with your actual API key

def gpt(factlist, model="gpt-5.2", num_example=1, dataset="wd50k-eval", data_dir="../data/"):
    client = OpenAI()
    answer = []
    for fact in factlist:
        prompt1, prompt2, prompt3, prompt4 = generate_prompt(fact, num_example, dataset, data_dir)
        messages = [
            {"role": "user", "content": prompt1},
            {"role": "assistant", "content": "Yes"},
            {"role": "user", "content": prompt2},
            {"role": "assistant", "content": "I have received your instructions and the goal statement"},
            {"role": "user", "content": prompt3},
            {"role": "assistant", "content": "I have analyzed the examples provided and will keep thinking without giving feedback."},
            {"role": "user", "content": prompt4}
        ]
        
        response = client.chat.completions.create(model=model, messages=messages)
        answer.append(response.choices[0].message.content)
    return answer

def parse_colon_format(line):
    parts = [part.strip() for part in line.split(':')]
    
    if len(parts) >= 3:
        return parts
    return None


def read_facts_from_file(filename):
    facts = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('['):
                try:
                    fact_list = ast.literal_eval(line)
                    
                    if isinstance(fact_list, list):
                        if all(isinstance(item, list) for item in fact_list):
                            facts.extend(fact_list)
                        else:
                            facts.append(fact_list)
                    continue
                except Exception as e:
                    print(f"Warning: line {line_num} parsing failed")
            
            parsed = parse_colon_format(line)
            facts.append(parsed)
    
    return facts


def save_results(results, output_file=None):
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"verification_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    


def run_verification_system(
    input_file, 
    model="gpt-5.2", 
    num_example=1, 
    dataset="wd50k-eval", 
    data_dir="../data/",
    output_file=None,
    batch_size=None
):
    facts = read_facts_from_file(input_file)
    
    if batch_size is None:
        batch_size = len(facts)
    
    all_results = []
    
    for i in range(0, len(facts), batch_size):
        batch = facts[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(facts) + batch_size - 1) // batch_size
        
        try:
            answers = gpt(batch, model, num_example, dataset, data_dir)
            
            for j, (fact, answer) in enumerate(zip(batch, answers)):
                result = {
                    "index": i + j,
                    "fact": fact,
                    "verification": answer
                }
                all_results.append(result)
                print(f"\nFact {i+j+1}")
                print(f"Input: {fact}")
                print(f"Result: {answer}")
        
        except Exception as e:
            print(e)
            continue
    
    final_results = {
        "total_facts": len(facts),
        "processed_facts": len(all_results),
        "model": model,
        "dataset": dataset,
        "timestamp": datetime.now().isoformat(),
        "results": all_results
    }
    
    save_results(final_results, output_file)
    
    return final_results
