import os
from pathlib import Path
from filter_fact import validate_facts
from gpt_verify import run_verification_system
import argparse
import json

def process_all_files(
    created_data_dir='./created_data',
    filtered_data_dir='./filtered_data',
    dataset_name='wd50k-eval',
    model='gpt-5.2',
    num_example=1,
    data_dir='../data/',
    batch_size=10
):
    os.makedirs(filtered_data_dir, exist_ok=True)
    
    created_data_path = Path(created_data_dir)
    if not created_data_path.exists():
        return
    
    txt_files = list(created_data_path.glob('*.txt'))
    
    if not txt_files:
        return
    
    for idx, input_file in enumerate(txt_files, 1):
        file_name = input_file.name
        
        filtered_file = os.path.join(filtered_data_dir, file_name)
        
        try:
            results = validate_facts(
                generated_facts_file=str(input_file),
                dataset_name=dataset_name,
                output_file=filtered_file
            )

            filter_result_file = filtered_file.replace('.txt', '_filter_stats.json')
            with open(filter_result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            continue
        
        if results['valid_facts'] == 0:
            continue
        
        verification_output = filtered_file.replace('.txt', '_verified.json')
        
        try:
            run_verification_system(
                input_file=filtered_file,
                model=model,
                num_example=num_example,
                dataset=dataset_name,
                data_dir=data_dir,
                output_file=verification_output,
                batch_size=batch_size
            )
        except Exception as e:
            print(e)
            continue
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process facts: filter and verify')
    parser.add_argument('--created_data_dir', default='./created_data')
    parser.add_argument('--filtered_data_dir', default='./filtered_data')
    parser.add_argument('--dataset', default='wd50k-eval')
    parser.add_argument('--model', default='gpt-5.2')
    parser.add_argument('--num_example', type=int, default=1)
    parser.add_argument('--data_dir', default='../data/')
    parser.add_argument('--batch_size', type=int, default=10)
    
    args = parser.parse_args()
    
    process_all_files(
        created_data_dir=args.created_data_dir,
        filtered_data_dir=args.filtered_data_dir,
        dataset_name=args.dataset,
        model=args.model,
        num_example=args.num_example,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )