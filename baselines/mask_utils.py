import os

def load_masked_facts(masked_file_path):
    masked_facts = []
    with open(masked_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                fact = eval(line)
                masked_facts.append(fact)
    return masked_facts

def load_train_facts(file_dir):
    train_file = os.path.join(file_dir, 'train.txt')
    train_facts = set()
    
    if not os.path.exists(train_file):
        return train_facts
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                fact = line.split('\t')
                train_facts.add(tuple(fact))
    
    return train_facts

def get_anchor_element(masked_fact):
    anchor_elements = []
    for idx, element in enumerate(masked_fact):
        if element != 'MASK':
            anchor_elements.append((element, idx))
    return anchor_elements

def get_anchor_element_with_idx(masked_fact):
    for idx, element in enumerate(masked_fact):
        if element != 'MASK':
            return element, idx
    return None, None

def is_entity_position(index):
    return index % 2 == 0


def has_mask(fact):
    """Check if a fact contains MASK token"""
    return 'MASK' in fact