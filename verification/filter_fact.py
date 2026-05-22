def parse_generated_facts(file_path):
    facts = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        if content.startswith('[[') and content.endswith(']]'):
            content = content[1:-1]
            
            current_fact = []
            in_fact = False
            current_item = ''
            in_quotes = False
            
            for char in content:
                if char in ('"', "'") and not in_quotes:
                    in_quotes = True
                elif char in ('"', "'") and in_quotes:
                    in_quotes = False
                    if current_item:
                        current_fact.append(current_item)
                        current_item = ''
                elif in_quotes:
                    current_item += char
                elif char == '[':
                    in_fact = True
                    current_fact = []
                elif char == ']':
                    in_fact = False
                    if current_fact:
                        facts.add(tuple(current_fact))
        else:
            for line in content.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                parts = line.strip('[]').split(',')
                parts = [p.strip().strip("'\"") for p in parts]
                
                if len(parts) > 0:
                    facts.add(tuple(parts))
    
    return facts


def load_entity_or_relation_set(file_path):
    items = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.add(line)
    return items


def load_fact_set(file_path):
    facts = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3 and len(parts) % 2 == 1:
                facts.add(tuple(parts))
    return facts


def validate_facts(generated_facts_file, dataset_name, output_file='filtered_facts.txt'):

    
    facts = parse_generated_facts(generated_facts_file)
    entities = load_entity_or_relation_set(f'{dataset_name}/entities.txt')
    relations = load_entity_or_relation_set(f'{dataset_name}/relations.txt')
    train_facts = load_fact_set(f'{dataset_name}/train.txt')
    valid_facts = load_fact_set(f'{dataset_name}/valid.txt')
    test_facts = load_fact_set(f'{dataset_name}/test.txt')
    

    invalid_entity_relation_count = 0
    in_train_count = 0
    in_test_count = 0
    valid_gens = []
    
    for fact in facts:
        if len(fact) < 3 or len(fact) % 2 == 0:
            invalid_entity_relation_count += 1
            continue
        
        is_valid = True
        
        for i, item in enumerate(fact):
            if i % 2 == 0:
                if item not in entities:
                    is_valid = False
                    break
            else:
                if item not in relations:
                    is_valid = False
                    break
        
        if not is_valid:
            invalid_entity_relation_count += 1
            continue
        
        if fact in train_facts:
            in_train_count += 1
            continue
        
        if fact in test_facts or fact in valid_facts:
            in_test_count += 1
            continue
        
        valid_gens.append(fact)

    with open(output_file, 'w', encoding='utf-8') as f:
        for fact in valid_gens:
            fact_str = "[" + ", ".join(f"'{item}'" for item in fact) + "]"
            f.write(f"{fact_str}\n")
    
    return {
        'invalid_entity_relation': invalid_entity_relation_count,
        'in_train': in_train_count,
        'in_test': in_test_count,
        'valid_facts': len(valid_gens)
    }