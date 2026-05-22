import os
import random
from collections import defaultdict
from crawl import convert_to_str

def safe_sample(population, k):
    if not population:
        return []
    actual_k = min(len(population), k)
    return random.sample(population, actual_k)

def generate_prompt(statements, num_example, dataset_name="wd50k-eval", data_dir="../data/"):
    dataset_dir = os.path.join(data_dir, dataset_name)+"/"
    for name in ["train.txt", "valid.txt", "test.txt"]:
        with open(dataset_dir + name) as f:
            facts = []
            for line in f.readlines():
                elements = line.strip().split()
                facts.append(elements)
    
    entity_index = defaultdict(list)
    relation_index = defaultdict(list)

    for fact in facts:
        h, r, t = fact[0], fact[1], fact[2]
        entity_index[h].append(fact)
        entity_index[t].append(fact)
        relation_index[r].append(fact)
        if len(fact) > 3:
            for i in range(3, len(fact), 2):
                q_rel = fact[i]
                q_val = fact[i+1]
                
                relation_index[q_rel].append(fact)
                entity_index[q_val].append(fact)

    full = []
    for i in range(len(statements)):
        if i % 2 == 0:
            full.append(safe_sample(entity_index[statements[i]], num_example))
        else:
            full.append(safe_sample(relation_index[statements[i]], num_example))
    statement_str = convert_to_str(statements,dataset_name)
    
    analogy = []
    supplement = []
    for i, statement in enumerate(full):
        for fact in statement:
            st = convert_to_str(fact,dataset_name)
            if st != statement_str:
                if i % 2 == 0:
                    supplement.append(st)
                else:
                    analogy.append(st)

    prompt1 = """
    You are an expert at verifying hyper-relational facts in a hyper-relational knowledge graph.
Given a hyper-relational fact, you need to verify whether the fact is (1) semantically valid and (2) factually true.
If you have understood your responsibility, respond 'Yes'. Otherwise, respond 'No'. Do not output anything except 'Yes' or 'No'.
"""

    prompt2 = f"""
    The goal statement is: "{statement_str}".
To verify the goal statement, you need to refer to other examples that may be similar or related to it.
Some of the given examples are similar to the goal statement. You should draw analogies from them to understand the potential meaning of the goal statement.
Other provided facts contain supplementary information; capture this extra information and mine potential relationships among them to support the verification.
Please carefully read, analyze, and reflect on these examples. Identify the reasoning patterns demonstrated in these examples and retain any information that may help your verification task.
While I provide examples, please remain silent until I ask you to respond.
"""

    prompt3 = "Examples used for analogy: "
    for i in analogy:
        prompt3 += f""" Verify the fact "{i}". The fact is semantically valid and factually true, so the verification result is Yes.""" 
    prompt3 += " Examples used to supplement information: "
    for i in supplement:
        prompt3 += f""" Verify the fact "{i}". The fact is semantically valid and factually true, so the verification result is Yes."""
    prompt3 += " Keep thinking, but DO NOT give me any feedback."

    prompt4 = f"""
The goal is to verify the fact "{statement_str}". Based on the previous examples and your own knowledge, provide a verification result.
- If the statement is semantically valid and factually true, respond "Yes".
- If the statement is semantically valid but factually false, respond "Half".
- If the statement is semantically invalid, respond "No".
Your answer must be consistent with all provided examples, even if it conflicts with standard linguistic definitions.

DO NOT OUTPUT ANYTHING ELSE.
    """
    return prompt1, prompt2, prompt3, prompt4