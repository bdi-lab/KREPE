def load_data(filename):
    data_map = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            
            if len(parts) >= 2:
                code = parts[0]
                content = parts[1]
                data_map[code] = content
        
    return data_map

def convert_to_str(fact, dataset, i=0):
    data_mape = load_data(f"../data/{dataset}/entities_labels.txt")
    data_mapr = load_data(f"../data/{dataset}/relations_labels.txt")
    
    if len(fact) == 1:
        if i%2==0:
            return data_mape.get(fact[0], fact[0])
        else:
            return data_mapr.get(fact[0], fact[0])

    head = data_mape.get(fact[0], fact[0])
    relation = data_mapr.get(fact[1], fact[1])
    tail = data_mape.get(fact[2], fact[2])
    st = f"{head} : {relation} : {tail}"
    if len(fact) > 3:
        for i in range(3, len(fact), 2):
            q_rel = fact[i]
            q_val = fact[i+1]
            q_rel_name = data_mapr.get(q_rel, q_rel)
            q_val_name = data_mape.get(q_val, q_val)
            st += f" : {q_rel_name} : {q_val_name}"
    return st
