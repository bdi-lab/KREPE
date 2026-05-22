import torch
import torch.nn.functional as F
from noise import LogLinearNoise

        
class DiffusionSampler():
    def __init__(self, model, batch_size, strategy, entity_para=0.5, relation_para=0.5, ent_temp=1, rel_temp=1, device=torch.device('cuda')):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.strategy = strategy
        self.entity_para = entity_para
        self.relation_para = relation_para
        self.ent_temp = ent_temp
        self.rel_temp = rel_temp
        self.eps = 1e-5
        self.noise = LogLinearNoise(eps=self.eps).to(self.device)
        self.method = 'tweedie'

    @torch.no_grad()
    def sample(self, steps, fact_length, query_pri, query_qual, query_qual2fact, \
                                    query_hpair, query_fact2hpair, \
                                    query_tpair, query_fact2tpair, \
                                    query_qpair, query_qual2qpair, \
                                    emb_ent, emb_rel, 
                                    ent_mask=None, rel_mask=None):
        self.model.eval()
        batch_size = query_pri.shape[0]
        ent_per_query = (fact_length + 1) // 2
        rel_per_query = fact_length // 2

        num_ent = emb_ent[0].shape[0]
        num_rel = emb_rel[0].shape[0]
        timesteps = torch.linspace(1, self.eps, steps + 1, device=self.device)
        
        if ent_mask is None:
            ent_mask = torch.ones(batch_size, ent_per_query, dtype=torch.bool, device=self.device)
        if rel_mask is None:
            rel_mask = torch.ones(batch_size, rel_per_query, dtype=torch.bool, device=self.device)
        
        changed = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        cached_pred_ent_logits = None
        cached_pred_rel_logits = None

        for i in range(steps):
            print(f"Diffusion sampling step {i+1}/{steps}", end='\r')
            if not ent_mask.any() and not rel_mask.any():
                break
            t = timesteps[i]
            update_rate = self.get_update_rate(t, steps)

            max_ent_cands = torch.max(query_pri[:, 0])
            max_ent_cands = max(max_ent_cands, torch.max(query_pri[:, 2]))
            max_ent_cands = max(max_ent_cands, num_ent - 1)
            if len(query_qual) > 0:
                max_ent_cands = max(max_ent_cands, torch.max(query_qual[:, 1]))
            num_ent_preds = max_ent_cands - num_ent + 1

            max_rel_cands = torch.max(query_pri[:, 1])
            max_rel_cands = max(max_rel_cands, num_rel - 1)
            if len(query_qual) > 0:
                max_rel_cands = max(max_rel_cands, torch.max(query_qual[:, 0]))
            num_rel_preds = max_rel_cands - num_rel + 1

            if cached_pred_ent_logits is None or cached_pred_ent_logits.shape[0] != num_ent_preds:
                cached_pred_ent_logits = torch.zeros(num_ent_preds, num_ent).cuda()
            if cached_pred_rel_logits is None or cached_pred_rel_logits.shape[0] != num_rel_preds:
                cached_pred_rel_logits = torch.zeros(num_rel_preds, num_rel).cuda()

            if changed.any():
                ent_pred, rel_pred = self.model.pred(query_pri.cuda(), query_qual.cuda(), 
                                    query_qual2fact.cuda(), \
                                    query_hpair.cuda(), query_fact2hpair.cuda(), \
                                    query_tpair.cuda(), query_fact2tpair.cuda(), \
                                    query_qpair.cuda(), query_qual2qpair.cuda(), \
                                    emb_ent, emb_rel, num_ent_preds, num_rel_preds)
                cached_pred_ent_logits = ent_pred
                cached_pred_rel_logits = rel_pred
            else:
                ent_pred = cached_pred_ent_logits
                rel_pred = cached_pred_rel_logits
            
            if i < steps - 1:
                ent_update_mask = ent_mask & (torch.rand(batch_size, ent_per_query, device=self.device) < update_rate)
                rel_update_mask = rel_mask & (torch.rand(batch_size, rel_per_query, device=self.device) < update_rate)
            else:
                ent_update_mask = ent_mask.clone()
                rel_update_mask = rel_mask.clone()
            query_pri_old = query_pri.clone()
            query_qual_old = query_qual.clone() if len(query_qual) > 0 else query_qual

            if ent_update_mask.any():
                target_ids = []
                target_update_info = [] 

                for b in range(batch_size):
                    if ent_update_mask[b, 0]:
                        target_ids.append(query_pri[b, 0].item())
                        target_update_info.append((0, b))
                    
                    if ent_update_mask[b, 1]:
                        target_ids.append(query_pri[b, 2].item())
                        target_update_info.append((1, b))
                    
                    qual_indices = (query_qual2fact == b).nonzero(as_tuple=True)[0]
                    
                    for j in range((fact_length - 2) // 2):
                        mask_idx = j + 2
                        if ent_update_mask[b, mask_idx]:
                            qual_idx = qual_indices[j]
                            target_ids.append(query_qual[qual_idx, 1].item())
                            target_update_info.append((2, qual_idx))
                
                if len(target_ids) > 0:
                    logit_indices = [eid - num_ent for eid in target_ids]
                    selected_logits = ent_pred[logit_indices] / self.ent_temp
                    
                    ent_update_values = sample_with_strategy(selected_logits, self.strategy, self.entity_para)

                    for idx, (update_type, table_idx) in enumerate(target_update_info):
                        new_val = ent_update_values[idx]
                        
                        if update_type == 0:
                            b = table_idx
                            query_pri[b, 0] = new_val
                            query_hpair[query_fact2hpair[b], 0] = new_val
                            
                        elif update_type == 1:
                            b = table_idx
                            query_pri[b, 2] = new_val
                            query_tpair[query_fact2tpair[b], 0] = new_val
                            
                        elif update_type == 2:
                            q_idx = table_idx
                            query_qual[q_idx, 1] = new_val
                            query_qpair[query_qual2qpair[q_idx], 1] = new_val

            if rel_update_mask.any():
                target_rel_ids = []
                target_rel_update_info = []
                
                for b in range(batch_size):
                    if rel_update_mask[b, 0]:
                        target_rel_ids.append(query_pri[b, 1].item())
                        target_rel_update_info.append((0, b))
                    
                    qual_indices = (query_qual2fact == b).nonzero(as_tuple=True)[0]
                    
                    for j in range((fact_length - 2) // 2):
                        mask_idx = j + 1
                        if rel_update_mask[b, mask_idx]:
                            qual_idx = qual_indices[j]
                            target_rel_ids.append(query_qual[qual_idx, 0].item())
                            target_rel_update_info.append((2, qual_idx))
                
                if len(target_rel_ids) > 0:
                    logit_indices = [rid - num_rel for rid in target_rel_ids]
                    selected_logits = rel_pred[logit_indices] / self.rel_temp
                    
                    rel_update_values = sample_with_strategy(selected_logits, self.strategy, self.relation_para)
                    
                    for idx, (update_type, table_idx) in enumerate(target_rel_update_info):
                        new_val = rel_update_values[idx]
                        
                        if update_type == 0: 
                            b = table_idx
                            query_pri[b, 1] = new_val
                            query_hpair[query_fact2hpair[b], 1] = new_val
                            query_tpair[query_fact2tpair[b], 1] = new_val
                        else:
                            q_idx = table_idx
                            query_qual[q_idx, 0] = new_val
                            query_qpair[query_qual2qpair[q_idx], 0] = new_val
            
            ent_mask = ent_mask & ~ent_update_mask
            rel_mask = rel_mask & ~rel_update_mask
            
            changed = (query_pri != query_pri_old).any(dim=-1)
            if len(query_qual) > 0:
                for b in range(batch_size):
                    qual_indices = (query_qual2fact == b).nonzero(as_tuple=True)[0]
                    if len(qual_indices) > 0:
                        qual_changed = (query_qual[qual_indices] != query_qual_old[qual_indices]).any()
                        changed[b] = changed[b] | qual_changed
            
        return query_pri, query_qual
    
    def get_update_rate(self, t, steps):
        dt = (1 - self.eps) / steps
        curr_sigma, next_sigma = self.noise(t)[0], self.noise(t - dt)[0]
        d_curr_sigma = self.noise(t)[1]
        if self.method == 'tweedie':
            update_rate = ((-next_sigma).exp() - (-curr_sigma).exp()) / (1 - (-curr_sigma).exp())
        elif self.method == 'euler':
            update_rate = dt * d_curr_sigma * (-curr_sigma).exp() / (1 - (-curr_sigma).exp())
        return update_rate
    

def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    
def direct_sampling(logits):
    probs = logits.softmax(dim=-1)
    index = sample_categorical(probs.to(torch.float32))
    return index


def top_p_sampling(logits, p=0.9):
    probs = logits.softmax(dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs.masked_fill_(indices_to_remove, 0)
    probs /= probs.sum(dim=-1).unsqueeze(-1)
    index = sample_categorical(probs.to(torch.float32))

    return index


def top_k_sampling(logits, k=10):
    top_k_values, top_k_indices = torch.topk(logits, int(k))
    top_k_probs = top_k_values.softmax(dim=-1)
    index = sample_categorical(top_k_probs.to(torch.float32))
    index = top_k_indices[torch.arange(index.size(0)), index]

    return index

def sample_with_strategy(update_logits, strategy, para = None):
    if strategy == "direct":
        return direct_sampling(update_logits)
    elif strategy == "top_p":
        return top_p_sampling(update_logits, para)
    elif strategy == "top_k":
        return top_k_sampling(update_logits, para)
    else:
        raise ValueError(f"Strategy {strategy} is not valid.")
