import os

def load_label_mappings(dataset_dir):
    id_to_entity_label = {}
    id_to_relation_label = {}
    
    entity_label_to_id = {}
    relation_label_to_id = {}
    
    entities_file = os.path.join(dataset_dir, 'entities_labels.txt')
    if os.path.exists(entities_file):
        with open(entities_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        entity_id, label = parts
                        id_to_entity_label[entity_id] = label
                        entity_label_to_id[label] = entity_id
    
    relations_file = os.path.join(dataset_dir, 'relations_labels.txt')
    if os.path.exists(relations_file):
        with open(relations_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        relation_id, label = parts
                        id_to_relation_label[relation_id] = label
                        relation_label_to_id[label] = relation_id
    
    return id_to_entity_label, id_to_relation_label, entity_label_to_id, relation_label_to_id


def load_facts(dataset_dir):
    facts = []
    for name in ["train.txt"]:
        path = os.path.join(dataset_dir, name)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    elements = line.strip('\t').split()
                    if len(elements) >= 3:
                        facts.append(elements)
    return facts


def convert_fact_to_natural_language(fact, id_to_entity, id_to_relation):
    nl_fact = []
    for i, elem in enumerate(fact):
        if i % 2 == 0:
            nl_fact.append(id_to_entity.get(elem, elem))
        else:
            nl_fact.append(id_to_relation.get(elem, elem))
    return nl_fact


def convert_fact_to_ids(fact, entity_to_id, relation_to_id):
    id_fact = []
    for i, elem in enumerate(fact):
        if i % 2 == 0:
            if elem in entity_to_id:
                id_fact.append(entity_to_id[elem])
            else:
                print(f"Warning: Entity '{elem}' not found in mapping")
                id_fact.append(elem)
        else:
            if elem in relation_to_id:
                id_fact.append(relation_to_id[elem])
            else:
                print(f"Warning: Relation '{elem}' not found in mapping")
                id_fact.append(elem)
    return id_fact

def load_entities_and_relations(dataset_dir):
    entities = []
    relations = []
    
    entities_file = os.path.join(dataset_dir, "entities.txt")
    if os.path.exists(entities_file):
        with open(entities_file, 'r', encoding='utf-8') as f:
            for line in f:
                entity = line.strip()
                if entity:
                    entities.append(entity)
    
    relations_file = os.path.join(dataset_dir, "relations.txt")
    if os.path.exists(relations_file):
        with open(relations_file, 'r', encoding='utf-8') as f:
            for line in f:
                relation = line.strip()
                if relation:
                    relations.append(relation)
    
    return entities, relations