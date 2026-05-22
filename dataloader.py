import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

class HKGDataset(Dataset):
    def __init__(self, datasets_dir, dataset_name, logger):
        self.dataset_dir = os.path.join(datasets_dir, dataset_name)+"/"
        self.name = dataset_name
        self.logger = logger


        logger.info(f"Loading {self.name}")

        self.train_graph = HKG(ref_graph = None)
        self.train_graph.parse_facts(self.dataset_dir, "train")
        self.train_graph.construct_inputs(device='cuda')

        self.valid_graph = HKG(ref_graph=self.train_graph)
        self.valid_graph.parse_facts(self.dataset_dir, "valid")
        self.test_graph = HKG(ref_graph=self.train_graph)
        self.test_graph.parse_facts(self.dataset_dir, "test")
        
        self.inference_graph = self.train_graph

        self.eval_filter_dict = {"valid":{}, "test":{}}
        self.construct_eval_filter_dict([self.inference_graph.facts, self.valid_graph.facts, self.test_graph.facts], mode = "valid")
        self.eval_filter_dict["test"] = self.eval_filter_dict["valid"]


        self.valid_graph.construct_query()
        self.test_graph.construct_query()
        logger.info(f"{self.name} Loaded.")
        logger.info(f"Train Graph: |V|={self.train_graph.stats['num_ent']}, |R|={self.train_graph.stats['num_rel']}, |H|={self.train_graph.num_fact}")
        logger.info(f"Valid Graph: |V|={self.valid_graph.stats['num_ent']}, |R|={self.valid_graph.stats['num_rel']}, |H|={self.valid_graph.num_fact}")
        logger.info(f"Test Graph: |V|={self.test_graph.stats['num_ent']}, |R|={self.test_graph.stats['num_rel']}, |H|={self.test_graph.num_fact}")

    def construct_eval_filter_dict(self, facts, mode = "valid"):
        for split in facts:
            for fact in split:
                for idx in range(len(fact)):
                    corrupted_fact = fact[:idx] + [-1] + fact[idx+1:]
                    primary_triplet = tuple(corrupted_fact[:3])
                    qualifiers = []
                    for i in range(len(corrupted_fact[3::2])):
                        qualifiers.append(tuple(corrupted_fact[3+2*i:5+2*i]))
                    filter_key_list = [primary_triplet]
                    if len(qualifiers) != 0:
                        filter_key_list += sorted(qualifiers)
                    filter_key = tuple(filter_key_list)
                    if filter_key not in self.eval_filter_dict[mode]:
                        self.eval_filter_dict[mode][filter_key] = []
                    self.eval_filter_dict[mode][filter_key].append(fact[idx])

    def train_split(self, base_idxs):
        g = self.train_graph 
        device = base_idxs.device
        base_idxs, _ = base_idxs.sort()
        seen_ent = torch.zeros(g.stats["num_ent"], dtype=torch.bool, device=device)
        seen_rel = torch.zeros(g.stats["num_rel"], dtype=torch.bool, device=device)
        base_mask = torch.zeros(len(g.pri), dtype=torch.bool, device=device)
        base_mask[base_idxs] = True

        seen_ent[g.pri[base_idxs, 0]] = True
        seen_ent[g.pri[base_idxs, 2]] = True
        if len(g.qual) > 0:
            subset_qual_mask = base_mask[g.qual2fact]
            seen_ent[g.qual[subset_qual_mask, 1]] = True
        unseen_ent = ~seen_ent

        seen_rel[g.pri[base_idxs, 1]] = True
        if len(g.qual) > 0:
            subset_qual_mask = base_mask[g.qual2fact]
            seen_rel[g.qual[subset_qual_mask, 0]] = True
        unseen_rel = ~seen_rel

        valid_ents = seen_ent[g.pri[:, 0]] & seen_ent[g.pri[:, 2]]
        valid_rels = seen_rel[g.pri[:, 1]]
        query_mask = valid_ents & valid_rels
        if len(g.qual) > 0:
            invalid_qual_ents = unseen_ent[g.qual[:, 1]]
            invalid_qual_rels = unseen_rel[g.qual[:, 0]]
            invalid_fact_idxs = g.qual2fact[invalid_qual_ents | invalid_qual_rels]
            query_mask[invalid_fact_idxs] = False
        query_mask[base_idxs] = False
        query_idxs = query_mask.nonzero(as_tuple=True)[0]

        pri = g.pri[base_idxs].clone().detach()
        valid_qual_idxs = base_mask[g.qual2fact].nonzero(as_tuple=True)[0]
        qual = g.qual[valid_qual_idxs].clone().detach()

        hpair = g.hpair.clone().detach()
        fact2hpair = g.fact2hpair.clone().detach()
        tpair = g.tpair.clone().detach()
        fact2tpair = g.fact2tpair.clone().detach()
        qpair = g.qpair.clone().detach()
        qual2qpair = g.qual2qpair.clone().detach()

        idx2idx = torch.zeros(len(g.pri), dtype=torch.long, device=device)
        idx2idx[base_idxs] = torch.arange(1, len(base_idxs) + 1, device=device)

        conv_ent = torch.full((g.stats["num_ent"],), -1, dtype=torch.long, device=device)
        conv_ent[seen_ent] = torch.arange(seen_ent.sum(), device=device)
        conv_rel = torch.full((g.stats["num_rel"],), -1, dtype=torch.long, device=device)
        conv_rel[seen_rel] = torch.arange(seen_rel.sum(), device=device)

        conv_pri = pri
        conv_pri[:, 0] = conv_ent[conv_pri[:, 0]]
        conv_pri[:, 1] = conv_rel[conv_pri[:, 1]]
        conv_pri[:, 2] = conv_ent[conv_pri[:, 2]]
        conv_qual2fact = idx2idx[g.qual2fact[valid_qual_idxs]]
        base_qual2fact = conv_qual2fact - 1
        conv_qual = qual
        if len(conv_qual) > 0:
            conv_qual[:, 0] = conv_rel[conv_qual[:, 0]]
            conv_qual[:, 1] = conv_ent[conv_qual[:, 1]]

        base_pri = conv_pri
        base_qual2fact = conv_qual2fact - 1
        base_qual = conv_qual


        base_hpair_freq = torch.bincount(fact2hpair[base_idxs], minlength = len(hpair))
        base_hpair = hpair[base_hpair_freq > 0]
        base_hpair_idx2idx = torch.full((len(hpair), ), -1, dtype = torch.long, device = device)
        base_hpair_idx2idx[base_hpair_freq > 0] = torch.arange(len(base_hpair), device = device)
        base_hpair_freq = base_hpair_freq[base_hpair_freq > 0]
        base_fact2hpair = base_hpair_idx2idx[fact2hpair[base_idxs]]
        base_hpair[:, 0] = conv_ent[base_hpair[:, 0]]
        base_hpair[:, 1] = conv_rel[base_hpair[:, 1]]

        base_tpair_freq = torch.bincount(fact2tpair[base_idxs], minlength = len(tpair))
        base_tpair = tpair[base_tpair_freq > 0]
        base_tpair_idx2idx = torch.full((len(tpair), ), -1, dtype = torch.long, device = device)
        base_tpair_idx2idx[base_tpair_freq > 0] = torch.arange(len(base_tpair), device = device)
        base_tpair_freq = base_tpair_freq[base_tpair_freq > 0]
        base_fact2tpair = base_tpair_idx2idx[fact2tpair[base_idxs]]
        base_tpair[:, 0] = conv_ent[base_tpair[:, 0]]
        base_tpair[:, 1] = conv_rel[base_tpair[:, 1]]

        base_qpair_freq = torch.bincount(qual2qpair[valid_qual_idxs], minlength = len(qpair))
        base_qpair = qpair[base_qpair_freq > 0]
        base_qpair_idx2idx = torch.full((len(qpair), ), -1, dtype = torch.long, device = device)
        base_qpair_idx2idx[base_qpair_freq > 0] = torch.arange(len(base_qpair), device = device)
        base_qpair_freq = base_qpair_freq[base_qpair_freq > 0]
        base_qual2qpair = base_qpair_idx2idx[qual2qpair[valid_qual_idxs]]
        if len(base_qpair) > 0:
            base_qpair[:, 0] = conv_rel[base_qpair[:, 0]]
            base_qpair[:, 1] = conv_ent[base_qpair[:, 1]]  

        num_base_ents = seen_ent.sum()
        num_base_rels = seen_rel.sum()
        
        return base_pri, base_qual, base_qual2fact, num_base_ents, num_base_rels, \
               base_hpair, base_fact2hpair, \
               base_tpair, base_fact2tpair, \
               base_qpair, base_qual2qpair, \
               conv_ent, conv_rel, query_idxs
    
    def train_preds(self, conv_ent, conv_rel, query_idxs):
        g = self.train_graph
        query_idxs, _ = query_idxs.sort()
        B = len(query_idxs)
        device = query_idxs.device
        fact_lengths = g.fact2len[query_idxs]
        max_len = torch.max(fact_lengths)
        seq_range = torch.arange(max_len, device=device)[None, :]

        valid_pos_mask = (seq_range < fact_lengths[:, None])

        rand_scores = torch.rand(B, max_len, device=device)
        rand_scores[~valid_pos_mask] = -1.0

        weights = valid_pos_mask.float()
        pivot_indices = torch.zeros(B, 1, device=device, dtype=torch.long)
        valid_rows_mask = weights.sum(dim=1) > 0
        pivot_indices[valid_rows_mask] = torch.multinomial(weights[valid_rows_mask], num_samples=1)
        tau_per_row = torch.gather(rand_scores, dim=1, index=pivot_indices)
        final_mask_2d = (rand_scores >= tau_per_row) & valid_pos_mask
        k_per_fact = final_mask_2d.sum(dim=1)
        
        nonzero_indices = final_mask_2d.nonzero()
        pred_idxs = query_idxs[nonzero_indices[:, 0]]
        pred_locs = nonzero_indices[:, 1]

        pred_ids = torch.zeros(len(pred_locs), dtype = torch.long, device = device)
        pred_ent_idxs = (pred_locs % 2 == 0).nonzero(as_tuple = True)[0]
        pred_rel_idxs = (pred_locs % 2 == 1).nonzero(as_tuple = True)[0]
        pred_ids[pred_ent_idxs] = torch.arange(len(pred_ent_idxs), device = device) + g.stats["num_ent"]
        pred_ids[pred_rel_idxs] = torch.arange(len(pred_rel_idxs), device = device) + g.stats["num_rel"]
        ent_answers = conv_ent[g.ent[g.fact2entstart[pred_idxs[pred_ent_idxs]] + pred_locs[pred_ent_idxs]//2]]
        rel_answers = conv_rel[g.rel[g.fact2relstart[pred_idxs[pred_rel_idxs]] + pred_locs[pred_rel_idxs]//2]]

        pri = g.pri[query_idxs].clone().detach()
        qual = g.qual.clone().detach()

        hpair = g.hpair.clone().detach()
        fact2hpair = g.fact2hpair.clone().detach()
        tpair = g.tpair.clone().detach()
        fact2tpair = g.fact2tpair.clone().detach()
        qpair = g.qpair.clone().detach()
        qual2qpair = g.qual2qpair.clone().detach()

        pri[nonzero_indices[pred_locs == 0, 0], 0] = pred_ids[pred_locs == 0]
        pri[nonzero_indices[pred_locs == 1, 0], 1] = pred_ids[pred_locs == 1]
        pri[nonzero_indices[pred_locs == 2, 0], 2] = pred_ids[pred_locs == 2]

        hpair_mask = (pred_locs == 0) | (pred_locs == 1)
        if hpair_mask.sum() > 0:
            hpair_fact_idxs = nonzero_indices[hpair_mask, 0]
            unique_hpair_fact_idxs = torch.unique(hpair_fact_idxs)
            fact2hpair[query_idxs[unique_hpair_fact_idxs]] = len(hpair) + torch.arange(unique_hpair_fact_idxs.shape[0], device = device)
            hpair = torch.cat([hpair, pri[unique_hpair_fact_idxs, :2]], dim = 0)

        tpair_mask = (pred_locs == 1) | (pred_locs == 2)
        if tpair_mask.sum() > 0:
            tpair_fact_idxs = nonzero_indices[tpair_mask,0]
            unique_tpair_fact_idxs = torch.unique(tpair_fact_idxs)
            fact2tpair[query_idxs[unique_tpair_fact_idxs]] = len(tpair) + torch.arange(unique_tpair_fact_idxs.shape[0], device = device)
            tpair = torch.cat([tpair, pri[unique_tpair_fact_idxs, 1:].flip(dims = (-1, ))], dim = 0)

        qual_mask = pred_locs > 2
        if qual_mask.sum() > 0:
            qual_pred_locs = g.fact2qualstart[pred_idxs][qual_mask] + (pred_locs[qual_mask] +1)//2 - 2
            if len(qual) > 0:
                qual[qual_pred_locs, (pred_locs[qual_mask] + 1) % 2] = pred_ids[qual_mask]
                unique_qual_locs = torch.unique(qual_pred_locs)
                qual2qpair[unique_qual_locs] = len(qpair) + torch.arange(unique_qual_locs.shape[0], device = device)
                qpair = torch.cat([qpair, qual[unique_qual_locs]], dim = 0)

        idx2idx = torch.zeros(g.num_fact, dtype = torch.long, device = device)
        idx2idx[query_idxs] = torch.arange(1, len(query_idxs) + 1, device = device)
        
        conv_ent = torch.cat([conv_ent, pred_ids[pred_ent_idxs] - g.stats["num_ent"] + (conv_ent != -1).sum()], dim = 0)
        conv_rel = torch.cat([conv_rel, pred_ids[pred_rel_idxs] - g.stats["num_rel"] + (conv_rel != -1).sum()], dim = 0)

        conv_pri = pri
        conv_pri[:, 0] = conv_ent[conv_pri[:, 0]]
        conv_pri[:, 1] = conv_rel[conv_pri[:, 1]]
        conv_pri[:, 2] = conv_ent[conv_pri[:, 2]]
        conv_qual2fact = idx2idx[g.qual2fact]
        conv_qual = qual
        if len(conv_qual) > 0:
            conv_qual[:, 0] = conv_rel[conv_qual[:, 0]]
            conv_qual[:, 1] = conv_ent[conv_qual[:, 1]]
        query_qual2fact_mask = (conv_qual2fact > 0)
        query_pri = conv_pri
        query_qual2fact = conv_qual2fact[query_qual2fact_mask] - 1 
        query_qual = conv_qual[query_qual2fact_mask]

        query_hpair_freq = torch.bincount(fact2hpair[query_idxs], minlength = len(hpair))
        query_hpair = hpair[query_hpair_freq > 0]
        query_hpair_idx2idx = torch.full((len(hpair), ), -1, dtype = torch.long, device = device)
        query_hpair_idx2idx[query_hpair_freq > 0] = torch.arange(len(query_hpair), device = device)
        query_hpair_freq = query_hpair_freq[query_hpair_freq > 0]
        query_fact2hpair = query_hpair_idx2idx[fact2hpair[query_idxs]]
        query_hpair[:, 0] = conv_ent[query_hpair[:, 0]]
        query_hpair[:, 1] = conv_rel[query_hpair[:, 1]]

        query_tpair_freq = torch.bincount(fact2tpair[query_idxs], minlength = len(tpair))
        query_tpair = tpair[query_tpair_freq > 0]
        query_tpair_idx2idx = torch.full((len(tpair), ), -1, dtype = torch.long, device = device)
        query_tpair_idx2idx[query_tpair_freq > 0] = torch.arange(len(query_tpair), device = device)
        query_tpair_freq = query_tpair_freq[query_tpair_freq > 0]
        query_fact2tpair = query_tpair_idx2idx[fact2tpair[query_idxs]]
        query_tpair[:, 0] = conv_ent[query_tpair[:, 0]]
        query_tpair[:, 1] = conv_rel[query_tpair[:, 1]]

        query_qpair_freq = torch.bincount(qual2qpair[conv_qual2fact > 0], minlength = len(qpair))
        query_qpair = qpair[query_qpair_freq > 0]
        query_qpair_idx2idx = torch.full((len(qpair), ), -1, dtype = torch.long, device = device)
        query_qpair_idx2idx[query_qpair_freq > 0] = torch.arange(len(query_qpair), device = device)
        query_qpair_freq = query_qpair_freq[query_qpair_freq > 0]
        query_qual2qpair = query_qpair_idx2idx[qual2qpair[conv_qual2fact > 0]]
        if len(query_qpair) > 0:
            query_qpair[:, 0] = conv_rel[query_qpair[:, 0]]
            query_qpair[:, 1] = conv_ent[query_qpair[:, 1]]
        return query_pri, query_qual, query_qual2fact, \
            query_hpair, query_fact2hpair, \
            query_tpair, query_fact2tpair, \
            query_qpair, query_qual2qpair, ent_answers, rel_answers, k_per_fact, nonzero_indices[:, 0], pred_locs, fact_lengths
    
    def eval_inputs(self, idxs, mode="valid"):
        if mode == "valid":
            target_queries = self.valid_graph.query 
        elif mode == "test":
            target_queries = self.test_graph.query
        else:
            raise NotImplementedError
        
        ref_graph = self.train_graph
        
        rel_idx = ref_graph.stats["num_rel"]
        ent_idx = ref_graph.stats["num_ent"]

        device = idxs.device
        pris = []
        qual2fact = []
        quals = []

        hpairs = []
        hpair2idx = {}
        hpair_freqs = []
        fact2hpairs = []

        tpairs = []
        tpair2idx = {}
        tpair_freqs = []
        fact2tpairs = []

        qpairs = []
        qpair2idx = {}
        qpair_freqs = []
        qual2qpairs = []

        ent_answers = []
        rel_answers = []

        ent_locs = []
        rel_locs = []

        ent_idxs = []
        rel_idxs = []

        for i, idx in enumerate(idxs):
            query = [list(comp) for comp in target_queries[idx]]
            pred_loc = sum(query, []).index(-1)
            pri = query[0]
            if pred_loc == 1:
                pri[1] = rel_idx
                rel_idx += 1
            elif pred_loc % 2 == 1:
                qual_idx = pred_loc // 2
                query[qual_idx][0] = rel_idx
                rel_idx += 1
            elif pred_loc == 0:
                pri[0] = ent_idx
                ent_idx += 1
            elif pred_loc == 2:
                pri[2] = ent_idx
                ent_idx += 1
            elif pred_loc % 2 == 0:
                qual_idx = pred_loc // 2 - 1
                query[qual_idx][1] = ent_idx
                ent_idx += 1
            else:
                raise NotImplementedError

            pris.append(pri)

            hpair = (pri[0], pri[1])
            tpair = (pri[2], pri[1])
            
            if hpair not in hpair2idx:
                hpair2idx[hpair] = len(hpairs)
                hpairs.append(list(hpair))
                hpair_freqs.append(0)
            fact2hpairs.append(hpair2idx[hpair])
            hpair_freqs[hpair2idx[hpair]] += 1
            
            if tpair not in tpair2idx:
                tpair2idx[tpair] = len(tpairs)
                tpairs.append(list(tpair))
                tpair_freqs.append(0)
            fact2tpairs.append(tpair2idx[tpair])
            tpair_freqs[tpair2idx[tpair]] += 1

            for qual in query[1:]:
                quals.append(qual)
                qual2fact.append(len(pris) - 1)
                qpair = tuple(qual)
                if qpair not in qpair2idx:
                    qpair2idx[qpair] = len(qpairs)
                    qpairs.append(list(qpair))
                    qpair_freqs.append(0)
                qual2qpairs.append(qpair2idx[qpair])
                qpair_freqs[qpair2idx[qpair]] += 1
            
            if pred_loc % 2 == 0:
                ent_answers.append(self.eval_filter_dict[mode][target_queries[idx]])
                ent_locs.append(pred_loc)
                ent_idxs.append(idx)
            else:
                rel_answers.append(self.eval_filter_dict[mode][target_queries[idx]])
                rel_locs.append(pred_loc)
                rel_idxs.append(idx)

        return torch.tensor(pris, dtype = torch.long, device = device), \
               torch.tensor(quals, dtype = torch.long, device = device).view(-1,2), \
               torch.tensor(qual2fact, dtype = torch.long, device = device), \
               torch.tensor(hpairs, dtype = torch.long, device = device), torch.tensor(fact2hpairs, dtype = torch.long, device = device), \
               torch.tensor(tpairs, dtype = torch.long, device = device), torch.tensor(fact2tpairs, dtype = torch.long, device = device), \
               torch.tensor(qpairs, dtype = torch.long, device = device).view(-1,2), torch.tensor(qual2qpairs, dtype = torch.long, device = device), \
               ent_answers, rel_answers, ent_locs, rel_locs, ent_idxs, rel_idxs
    
    def sample_query(self, batch_size, fact_length, device = None):

        assert fact_length >= 3 and fact_length % 2 == 1, "fact_length must be odd and >= 3"
    
        B = batch_size
        max_len = fact_length
        

        batch_indices = torch.arange(B, device=device).repeat_interleave(fact_length)
        position_indices = torch.arange(fact_length, device=device).repeat(B)
        
        pred_idxs = batch_indices
        pred_locs = position_indices
        
        pred_ids = torch.zeros(len(pred_locs), dtype=torch.long, device=device)
        pred_ent_mask = (pred_locs % 2 == 0)
        pred_rel_mask = (pred_locs % 2 == 1)
        
        pred_ids[pred_ent_mask] = torch.arange(pred_ent_mask.sum(), device=device) + self.inference_graph.stats['num_ent']
        pred_ids[pred_rel_mask] = torch.arange(pred_rel_mask.sum(), device=device) + self.inference_graph.stats['num_rel']
        
        pri = torch.zeros(B, 3, dtype=torch.long, device=device)
        pri[pred_idxs[pred_locs == 0], 0] = pred_ids[pred_locs == 0]
        pri[pred_idxs[pred_locs == 1], 1] = pred_ids[pred_locs == 1] 
        pri[pred_idxs[pred_locs == 2], 2] = pred_ids[pred_locs == 2] 
        
        qual_mask = pred_locs > 2
        num_quals = 0
        qual = torch.empty(0, 2, dtype=torch.long, device=device)
        qual2fact = torch.empty(0, dtype=torch.long, device=device)
        
        if qual_mask.sum() > 0:
            qual_fact_idxs = pred_idxs[qual_mask]
            qual_pos_in_fact = pred_locs[qual_mask]
            
            qual_indices = (qual_pos_in_fact - 3) // 2
            
            max_qual_idx = qual_indices.max().item() + 1
            num_quals = len(torch.unique(qual_fact_idxs * max_qual_idx + qual_indices))
            
            qual = torch.zeros(num_quals, 2, dtype=torch.long, device=device)
            qual2fact = torch.zeros(num_quals, dtype=torch.long, device=device)
            
            unique_qual_keys = qual_fact_idxs * max_qual_idx + qual_indices
            unique_quals, inverse_indices = torch.unique(unique_qual_keys, return_inverse=True)
            
            for i, (fact_idx, pos) in enumerate(zip(qual_fact_idxs, qual_pos_in_fact)):
                qual_id = inverse_indices[i]
                qual2fact[qual_id] = fact_idx
                
                if pos % 2 == 1:
                    qual[qual_id, 0] = pred_ids[qual_mask][i]
                else:
                    qual[qual_id, 1] = pred_ids[qual_mask][i]
        
        hpair_mask = (pred_locs == 0) | (pred_locs == 1)
        fact2hpair = torch.zeros(B, dtype=torch.long, device=device)
        
        if hpair_mask.sum() > 0:
            hpair_fact_idxs = pred_idxs[hpair_mask]
            unique_hpair_fact_idxs = torch.unique(hpair_fact_idxs)
            fact2hpair[unique_hpair_fact_idxs] = torch.arange(len(unique_hpair_fact_idxs), device=device)
            hpair = pri[unique_hpair_fact_idxs, :2]
        else:
            hpair = torch.empty(0, 2, dtype=torch.long, device=device)
        
        tpair_mask = (pred_locs == 1) | (pred_locs == 2)
        fact2tpair = torch.zeros(B, dtype=torch.long, device=device)
        
        if tpair_mask.sum() > 0:
            tpair_fact_idxs = pred_idxs[tpair_mask]
            unique_tpair_fact_idxs = torch.unique(tpair_fact_idxs)
            fact2tpair[unique_tpair_fact_idxs] = torch.arange(len(unique_tpair_fact_idxs), device=device)
            tpair = pri[unique_tpair_fact_idxs, 1:].flip(dims=(-1,))
        else:
            tpair = torch.empty(0, 2, dtype=torch.long, device=device)
        
        qual2qpair = torch.zeros(num_quals, dtype=torch.long, device=device)
        
        if num_quals > 0:
            qual2qpair = torch.arange(num_quals, device=device)
            qpair = qual.clone()
        else:
            qpair = torch.empty(0, 2, dtype=torch.long, device=device)

        return pri.cuda(), qual.cuda(), qual2fact.cuda(),\
                hpair.cuda(), fact2hpair.cuda(),\
                tpair.cuda(), fact2tpair.cuda(),\
                qpair.cuda(), qual2qpair.cuda()
    
    def generate_and_mask_facts(self, facts, location, entities_or_rels=None, m=1, n=5):

        device = self.train_graph.pri.device
        if facts is None:
            facts = []
            for _ in range(m):
                fact = []
                for idx in range(n):
                    if idx % 2 == 0:
                        random_ent_id = random.randint(0, self.train_graph.stats['num_ent'] - 1)
                        fact.append(random_ent_id)
                    else:
                        random_rel_id = random.randint(0, self.train_graph.stats['num_rel'] - 1)
                        fact.append(random_rel_id)
                facts.append(fact)
        
        if entities_or_rels is not None and location >= 0:
            is_iterable = isinstance(entities_or_rels, (list, torch.Tensor, np.ndarray))
            
            for i, fact in enumerate(facts):
                if is_iterable:
                    replace_id = entities_or_rels[i]
                    if isinstance(replace_id, torch.Tensor):
                        replace_id = replace_id.item()
                else:
                    replace_id = entities_or_rels
                
                fact[location] = replace_id

        mask_loc = location + 1
        if mask_loc < n:
            mask_ent_id = self.train_graph.stats['num_ent']
            mask_rel_id = self.train_graph.stats['num_rel']
            
            for i, fact in enumerate(facts):
                if mask_loc % 2 == 0:
                    fact[mask_loc] = mask_ent_id + i
                else:
                    fact[mask_loc] = mask_rel_id + i
        
        pris = []
        quals = []
        qual2fact = []
        
        hpairs = []
        hpair2idx = {}
        fact2hpairs = []
        
        tpairs = []
        tpair2idx = {}
        fact2tpairs = []
        
        qpairs = []
        qpair2idx = {}
        qual2qpairs = []
        
        for fact_idx, fact in enumerate(facts):
            pris.append([fact[0], fact[1], fact[2]])
            
            hpair = (fact[0], fact[1])
            if hpair not in hpair2idx:
                hpair2idx[hpair] = len(hpairs)
                hpairs.append(list(hpair))
            fact2hpairs.append(hpair2idx[hpair])
            
            tpair = (fact[2], fact[1])
            if tpair not in tpair2idx:
                tpair2idx[tpair] = len(tpairs)
                tpairs.append(list(tpair))
            fact2tpairs.append(tpair2idx[tpair])
            
            for i in range(3, len(fact), 2):
                if i + 1 < len(fact):
                    qual_rel = fact[i]
                    qual_ent = fact[i + 1]
                    quals.append([qual_rel, qual_ent])
                    qual2fact.append(fact_idx)
                    
                    qpair = (qual_rel, qual_ent)
                    if qpair not in qpair2idx:
                        qpair2idx[qpair] = len(qpairs)
                        qpairs.append(list(qpair))
                    qual2qpairs.append(qpair2idx[qpair])
        
        query_pri = torch.tensor(pris, dtype=torch.long, device=device)
        query_qual = torch.tensor(quals, dtype=torch.long, device=device).view(-1, 2) if quals else torch.empty(0, 2, dtype=torch.long, device=device)
        query_qual2fact = torch.tensor(qual2fact, dtype=torch.long, device=device) if qual2fact else torch.empty(0, dtype=torch.long, device=device)
        
        query_hpair = torch.tensor(hpairs, dtype=torch.long, device=device)
        query_fact2hpair = torch.tensor(fact2hpairs, dtype=torch.long, device=device)
        
        query_tpair = torch.tensor(tpairs, dtype=torch.long, device=device)
        query_fact2tpair = torch.tensor(fact2tpairs, dtype=torch.long, device=device)
        
        query_qpair = torch.tensor(qpairs, dtype=torch.long, device=device).view(-1, 2) if qpairs else torch.empty(0, 2, dtype=torch.long, device=device)
        query_qual2qpair = torch.tensor(qual2qpairs, dtype=torch.long, device=device) if qual2qpairs else torch.empty(0, dtype=torch.long, device=device)
        
        return query_pri, query_qual, query_qual2fact, \
            query_hpair, query_fact2hpair, \
            query_tpair, query_fact2tpair, \
            query_qpair, query_qual2qpair
    
class HKG(Dataset):
    def __init__(self,ref_graph = None):

        if ref_graph is None:
            self.ent2id = {}
            self.id2ent = []
            self.stats = {"num_ent": 0, "num_rel": 0}
            self.rel2id = {}
            self.id2rel = []
            self.new = True
        else:
            self.ent2id = ref_graph.ent2id
            self.id2ent = ref_graph.id2ent
            self.stats = ref_graph.stats
            self.rel2id = ref_graph.rel2id
            self.id2rel = ref_graph.id2rel
            self.new = True

        self.facts = []

        self.pri = []
        self.qual = []
        self.qual2fact = []

        self.rel = []
        self.rel2fact = []
        self.relloc = []
        self.fact2len = []
        self.fact2numrel = []
        self.fact2qualstart = []
        self.fact2relstart = []
        self.fact2rellocstart = []

        self.ent = []
        self.ent2fact = []
        self.entloc = []
        self.fact2nument = []
        self.fact2entstart = []
        self.fact2entlocstart = []

        self.hpair = []
        self.tpair = []
        self.qpair = []
        self.fact2hpair = []
        self.fact2tpair = []
        self.qual2qpair = []

        
    def parse_facts(self, dataset_dir, mode):
        with open(dataset_dir + f"{mode}.txt") as f:
            for line in f.readlines():
                fact = []
                elements = line.strip().split("\t")
                for idx, token in enumerate(elements):
                    is_entity = (idx % 2 == 0)
                    curr_map = self.ent2id if is_entity else self.rel2id
                    curr_list = self.id2ent if is_entity else self.id2rel
                    if token not in curr_map and self.new:
                        curr_map[token] = len(curr_list)
                        curr_list.append(token)
                        if is_entity:
                            self.stats["num_ent"] += 1
                        else:
                            self.stats["num_rel"] += 1
                    fact.append(curr_map[token])
                self.facts.append(fact)
        self.num_fact = len(self.facts)

    def construct_query(self):
        query2idx = {}
        split_query = []
        split_answer = []

        for fact in self.facts:
            for idx in range(len(fact)):
                corrupted_fact = fact[:idx] + [-1] + fact[idx+1:]
                primary_triplet = tuple(corrupted_fact[:3])
                qualifiers = []
                for i in range(len(corrupted_fact[3::2])):
                    qualifiers.append(tuple(corrupted_fact[3+2*i:5+2*i]))
                filter_key_list = [primary_triplet]
                if len(qualifiers) != 0:
                    filter_key_list += sorted(qualifiers)
                filter_key = tuple(filter_key_list)
                if filter_key not in query2idx:
                    query2idx[filter_key] = len(split_query)
                    split_query.append(filter_key)
                    split_answer.append([fact[idx]])
                else:
                    loc = query2idx[filter_key]
                    split_answer[loc].append(fact[idx])
        split_answer = [a for _, a in sorted(zip(split_query, split_answer))]
        split_query = sorted(split_query)
        self.query, self.answer = split_query, split_answer

    def construct_inputs(self, device = 'cuda'):
        hpair2idx = {}
        tpair2idx = {}
        qpair2idx = {}
        
        for idx, fact in enumerate(self.facts):
            self.pri.append([fact[0], fact[1], fact[2]])

            self.fact2len.append(len(fact))
            self.fact2numrel.append(0)
            self.fact2nument.append(0)
            
            self.fact2rellocstart.append(len(self.relloc))
            self.fact2qualstart.append(len(self.qual))
            self.fact2relstart.append(len(self.rel))
            self.fact2entlocstart.append(len(self.entloc))
            self.fact2entstart.append(len(self.ent))
            
            hpair = (fact[0], fact[1])
            tpair = (fact[2], fact[1])
            
            if hpair not in hpair2idx:
                hpair2idx[hpair] = len(self.hpair)
                self.hpair.append(list(hpair))
            if tpair not in tpair2idx:
                tpair2idx[tpair] = len(self.tpair)
                self.tpair.append(list(tpair))
                
            self.fact2hpair.append(hpair2idx[hpair])
            self.fact2tpair.append(tpair2idx[tpair])
            
            for i, _ in enumerate(fact):
                if i > 2 and i % 2 == 0:
                    qpair = (fact[i-1], fact[i])
                    self.qual2fact.append(idx)
                    self.qual.append(list(qpair))
                    
                    if qpair not in qpair2idx:
                        qpair2idx[qpair] = len(self.qpair)
                        self.qpair.append(list(qpair))
                    self.qual2qpair.append(qpair2idx[qpair])
                
                if i % 2 == 0:
                    self.fact2nument[-1] += 1
                    self.entloc.append(i)
                    self.ent.append(fact[i])
                    self.ent2fact.append(idx)
                else:
                    self.fact2numrel[-1] += 1
                    self.relloc.append(i)
                    self.rel.append(fact[i])
                    self.rel2fact.append(idx)

                
        self.pri = torch.tensor(self.pri, dtype = torch.long, device = device)
        self.qual2fact = torch.tensor(self.qual2fact, dtype = torch.long, device = device)
        self.qual = torch.tensor(self.qual, dtype = torch.long, device = device).view(-1,2)
        self.rel = torch.tensor(self.rel, dtype = torch.long, device = device)
        self.rel2fact = torch.tensor(self.rel2fact, dtype = torch.long, device = device)
        self.relloc = torch.tensor(self.relloc, dtype = torch.long, device = device)
        self.fact2len = torch.tensor(self.fact2len, dtype = torch.long, device = device)
        self.fact2numrel = torch.tensor(self.fact2numrel, dtype = torch.long, device = device)
        self.fact2relstart = torch.tensor(self.fact2relstart, dtype = torch.long, device = device)
        self.fact2qualstart = torch.tensor(self.fact2qualstart, dtype = torch.long, device = device)
        self.fact2rellocstart = torch.tensor(self.fact2rellocstart, dtype = torch.long, device = device)

        self.ent = torch.tensor(self.ent, dtype = torch.long, device = device)
        self.ent2fact = torch.tensor(self.ent2fact, dtype = torch.long, device = device)
        self.entloc = torch.tensor(self.entloc, dtype = torch.long, device = device)
        self.fact2nument = torch.tensor(self.fact2nument, dtype = torch.long, device = device)
        self.fact2entstart = torch.tensor(self.fact2entstart, dtype = torch.long, device = device)
        self.fact2entlocstart = torch.tensor(self.fact2entlocstart, dtype = torch.long, device = device)

        self.hpair = torch.tensor(self.hpair, dtype = torch.long, device = device)
        self.fact2hpair = torch.tensor(self.fact2hpair, dtype = torch.long, device = device)
        self.tpair = torch.tensor(self.tpair, dtype = torch.long, device = device)
        self.fact2tpair = torch.tensor(self.fact2tpair, dtype = torch.long, device = device)
        self.qpair = torch.tensor(self.qpair, dtype = torch.long, device = device).view(-1,2)
        self.qual2qpair = torch.tensor(self.qual2qpair, dtype = torch.long, device = device)
        torch.cuda.empty_cache()