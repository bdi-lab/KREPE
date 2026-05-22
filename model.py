import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Union, Annotated
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint

class FFN(nn.Module):
    def __init__(self, act: str = "ReLU", dim_in: int = 32, dim_hidden: int = 32, dim_out: int = 32, dropout_rate: float = 0.2):
        super(FFN, self).__init__()

        if act == "ReLU":
            self.act = nn.ReLU()
        elif act == "LeakyReLU":
            self.act = nn.LeakyReLU()
        elif act == "PReLU":
            self.act = nn.PReLU()
        elif act == "GELU":
            self.act = nn.GELU()
        elif act == "tanh":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError

        self.net = nn.Sequential(nn.Linear(dim_in, dim_hidden),
                                 self.act,
                                 nn.Dropout(p = dropout_rate),
                                 nn.Linear(dim_hidden, dim_out))
        self.ln = nn.LayerNorm(dim_in)
        self.param_init()

    def param_init(self):
        nn.init.kaiming_normal_(self.net[0].weight, mode = 'fan_out', nonlinearity = 'relu')
        nn.init.zeros_(self.net[0].bias)

        nn.init.xavier_uniform_(self.net[3].weight, gain = 0.01)
        nn.init.zeros_(self.net[3].bias)
    
    def forward(self, x):
        return self.net(self.ln(x))

class attention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_head, dropout_rate):
        super(attention, self).__init__()
        self.dim_q = dim_q
        self.num_head = num_head
        self.dim_per_head = dim_q // num_head
        self.scale = 1.0/math.sqrt(self.dim_per_head)
        self.drop = nn.Dropout(p = dropout_rate)

        self.q_QKV = nn.Linear(dim_q, 3*dim_q, bias = True)
        self.kv_KV = nn.Linear(dim_kv, 2*dim_q, bias = True)

        self.P = nn.Linear(dim_q, dim_q, bias = True)

        self.param_init()
    
    def param_init(self):
        for w in [self.q_QKV, self.kv_KV]:
            nn.init.xavier_uniform_(w.weight)
            if w.bias is not None:
                nn.init.zeros_(w.bias)

        nn.init.xavier_uniform_(self.P.weight, gain = 0.01)
        if self.P.bias is not None:
            nn.init.zeros_(self.P.bias)

    def forward(self, query, kv, query_idx):
        device = query.device

        q_query, k_query, v_query = self.q_QKV(query).chunk(3, dim = -1)
        k_kv, v_kv = self.kv_KV(kv).chunk(2, dim = -1)

        attn_self = (q_query * k_query).view(-1, self.num_head, self.dim_per_head).sum(dim = -1, keepdim = True) * self.scale
        attn_self = attn_self.float()

        attn_nb = (torch.index_select(q_query, 0, query_idx) * k_kv).view(-1, self.num_head, self.dim_per_head).sum(dim = -1, keepdim = True) * self.scale
        attn_nb = attn_nb.float()

        with torch.no_grad():
            max_attn = attn_self.index_reduce(dim = 0, index = query_idx, source = attn_nb, reduce = "amax", include_self = True)

        exp_attn_self = torch.exp(attn_self - max_attn)
            
        exp_attn_nb = torch.exp(attn_nb - max_attn[query_idx])
        sum_exp_attn = exp_attn_self.index_add(dim = 0, index = query_idx, source = exp_attn_nb)

        attn_weight_self = exp_attn_self / sum_exp_attn
        attn_weight_nb = self.drop(exp_attn_nb / torch.index_select(sum_exp_attn, 0, query_idx))

        msg_to_query = attn_weight_self * v_query.view(-1, self.num_head, self.dim_per_head)

        msg_to_query = msg_to_query.index_add(dim = 0, index = query_idx, source = attn_weight_nb * v_kv.view(-1, self.num_head, self.dim_per_head))

        msg_to_query = msg_to_query.flatten(start_dim = 1)
        msg_to_query = msg_to_query.to(query.dtype)
        new_query = self.P(msg_to_query)

        return new_query

class KREPE_layer(nn.Module):
    def __init__(self, act, dim, num_head_ent, num_head_rel, dropout = 0.2):
        super(KREPE_layer, self).__init__()
        self.dim = dim
        self.drop = nn.Dropout(p = dropout)

        self.proj_ent_base = nn.Linear(dim, 4 * dim, bias = True)
        self.proj_rel_base = nn.Linear(dim, 4 * dim, bias = True)

        self.proj_fact = nn.Linear(dim, 2 * dim, bias = True)

        self.norm_h = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)

        self.norm_ent = nn.LayerNorm(dim)
        self.norm_rel = nn.LayerNorm(dim)

        self.mlp_fact = FFN(act, dim, dim, dim, dropout)
        self.mlp_ent = FFN(act, dim, dim, dim, dropout)
        self.mlp_rel = FFN(act, dim, dim, dim, dropout)

        self.ent_attn = attention(dim, dim, num_head_ent, dropout)
        self.rel_attn = attention(dim, dim, num_head_rel, dropout)

        self.param_init()

    def param_init(self):
        for w in [self.proj_ent_base, self.proj_rel_base, self.proj_fact]:
            nn.init.kaiming_normal_(w.weight, mode = 'fan_out', nonlinearity = 'relu')
            nn.init.zeros_(w.bias)

    def _compute_messages(self, emb_ent, emb_rel, pri, qual2fact, qual,\
                          hpair_ent, hpair_rel, fact2hpair,
                          tpair_ent, tpair_rel, fact2tpair,
                          qpair_ent, qpair_rel, qual2qpair):
        
        heads, rels, tails = pri[:,0], pri[:,1], pri[:,2]
        qual_rels, qual_ents = qual[:,0], qual[:,1]
        num_pri = len(pri)
        num_qual = len(qual)

        he2f, te2f, qe2f, fe2r = self.proj_ent_base(emb_ent).chunk(4, dim = -1)
        hr2f, tr2f, qr2f, fr2e = self.proj_rel_base(emb_rel).chunk(4, dim = -1)

        emb_hpair = torch.index_select(he2f, 0, hpair_ent) + torch.index_select(hr2f, 0, hpair_rel)
        emb_tpair = torch.index_select(te2f, 0, tpair_ent) + torch.index_select(tr2f, 0, tpair_rel)
        emb_qpair = torch.index_select(qe2f, 0, qpair_ent) + torch.index_select(qr2f, 0, qpair_rel)

        new_emb_fact = torch.index_select(emb_hpair, 0, fact2hpair) + torch.index_select(emb_tpair, 0, fact2tpair)
        new_emb_fact = new_emb_fact.index_add(0, qual2fact, torch.index_select(emb_qpair, 0, qual2qpair))
        f2cnt = (torch.bincount(qual2fact, minlength = num_pri) + 1).unsqueeze(dim = 1)
        inv_cnt = 1.0 / f2cnt

        f2h = (new_emb_fact - torch.index_select(emb_hpair, 0, fact2hpair)) * inv_cnt
        f2h_norm = self.norm_h(f2h)
        f2t = (new_emb_fact - torch.index_select(emb_tpair, 0, fact2tpair)) * inv_cnt
        f2t_norm = self.norm_t(f2t)
        f2q = (torch.index_select(new_emb_fact, 0, qual2fact) - torch.index_select(emb_qpair, 0, qual2qpair)) * inv_cnt[qual2fact]
        if num_qual > 0:
            f2q_norm = self.norm_q(f2q)
        else:
            f2q_norm = torch.empty((0, self.dim), dtype = f2h_norm.dtype, device = f2h_norm.device)
        f2s = torch.cat([f2h_norm, f2t_norm, f2q_norm], dim = 0)
        f2s = f2s + self.drop(self.mlp_fact(f2s))
        f2e, f2r = self.proj_fact(f2s).chunk(2, dim = -1)
        f2he, f2te, f2qe = f2e.split([num_pri, num_pri, num_qual], dim = 0)
        f2hr, f2tr, f2qr = f2r.split([num_pri, num_pri, num_qual], dim = 0)

        return (f2he, f2te, f2qe, fr2e), (f2hr, f2tr, f2qr, fe2r), (heads, rels, tails, qual_rels, qual_ents)

    def forward(self, emb_ent, emb_rel, \
                pri, qual2fact, qual, \
                hpair_ent, hpair_rel, fact2hpair, \
                tpair_ent, tpair_rel, fact2tpair, \
                qpair_ent, qpair_rel, qual2qpair):

        emb_ent_norm = self.norm_ent(emb_ent)
        emb_rel_norm = self.norm_rel(emb_rel)
        
        ent_msgs, rel_msgs, indices = self._compute_messages(emb_ent_norm, emb_rel_norm, pri, qual2fact, qual,
                                                                        hpair_ent, hpair_rel, fact2hpair,
                                                                        tpair_ent, tpair_rel, fact2tpair,
                                                                        qpair_ent, qpair_rel, qual2qpair)
        
        f2he, f2te, f2qe, fr2e = ent_msgs
        f2hr, f2tr, f2qr, fe2r = rel_msgs
        heads, rels, tails, qual_rels, qual_ents = indices

        f2he = f2he + torch.index_select(fr2e, 0, rels)
        f2te = f2te + torch.index_select(fr2e, 0, rels)
        f2qe = f2qe + torch.index_select(fr2e, 0, qual_rels)

        kv_ent = torch.cat([f2he, f2te, f2qe], dim = 0)
        upd_ent = self.ent_attn(query = emb_ent_norm, kv = kv_ent, query_idx = torch.cat([heads, tails, qual_ents], dim = 0))
        upd_ent = emb_ent + self.drop(upd_ent)
        new_emb_ent = upd_ent + self.drop(self.mlp_ent(upd_ent))

        f2hr = f2hr + torch.index_select(fe2r, 0, heads)
        f2tr = f2tr + torch.index_select(fe2r, 0, tails)
        f2qr = f2qr + torch.index_select(fe2r, 0, qual_ents)

        kv_rel = torch.cat([f2hr, f2tr, f2qr], dim = 0)
        upd_rel = self.rel_attn(query = emb_rel_norm, kv = kv_rel, query_idx = torch.cat([rels, rels, qual_rels], dim = 0))
        upd_rel = emb_rel + self.drop(upd_rel)
        new_emb_rel = upd_rel + self.drop(self.mlp_rel(upd_rel))

        return new_emb_ent , new_emb_rel
    
    def pred(self, num_ent, num_rel, emb_ent, emb_rel, pri, qual2fact, qual, \
             hpair_ent, hpair_rel, fact2hpair, \
             tpair_ent, tpair_rel, fact2tpair, \
             qpair_ent, qpair_rel, qual2qpair):
        
        device = emb_ent.device

        emb_ent_norm = self.norm_ent(emb_ent)
        emb_rel_norm = self.norm_rel(emb_rel)

        ent_msgs, rel_msgs, indices = self._compute_messages(emb_ent_norm, emb_rel_norm, pri, qual2fact, qual,
                                                                        hpair_ent, hpair_rel, fact2hpair,
                                                                        tpair_ent, tpair_rel, fact2tpair,
                                                                        qpair_ent, qpair_rel, qual2qpair)

        f2he, f2te, f2qe, fr2e = ent_msgs
        f2hr, f2tr, f2qr, fe2r = rel_msgs
        heads, rels, tails, qual_rels, qual_ents = indices

        pred_heads = (heads >= num_ent)
        pred_tails = (tails >= num_ent)
        pred_qual_ents = (qual_ents >= num_ent)

        f2he = f2he[pred_heads] + torch.index_select(fr2e, 0, rels[pred_heads])
        f2te = f2te[pred_tails] + torch.index_select(fr2e, 0, rels[pred_tails])
        f2qe = f2qe[pred_qual_ents] + torch.index_select(fr2e, 0, qual_rels[pred_qual_ents])


        pred_ent_idxs = torch.arange(num_ent, len(emb_ent), device = device)
        kv_ent = torch.cat([f2he, f2te, f2qe], dim = 0)
        upd_ent = self.ent_attn(query = emb_ent_norm[num_ent:], kv = kv_ent, query_idx = torch.cat([heads[pred_heads], tails[pred_tails], qual_ents[pred_qual_ents]], dim = 0) - num_ent)
        upd_ent = emb_ent[num_ent:] + self.drop(upd_ent)
        new_pred_emb_ent = upd_ent + self.drop(self.mlp_ent(upd_ent))

        pred_pri_rels = (rels >= num_rel)
        pred_qual_rels = (qual_rels >= num_rel)

        f2hr = f2hr[pred_pri_rels] + torch.index_select(fe2r, 0, heads[pred_pri_rels])
        f2tr = f2tr[pred_pri_rels] + torch.index_select(fe2r, 0, tails[pred_pri_rels])
        f2qr = f2qr[pred_qual_rels] + torch.index_select(fe2r, 0, qual_ents[pred_qual_rels])

        pred_rel_idxs = torch.arange(num_rel, len(emb_rel), device = device)
        kv_rel = torch.cat([f2hr, f2tr, f2qr], dim = 0)
        upd_rel = self.rel_attn(query = emb_rel_norm[num_rel:], kv = kv_rel, query_idx = torch.cat([rels[pred_pri_rels], rels[pred_pri_rels], qual_rels[pred_qual_rels]], dim = 0) - num_rel)
        upd_rel = emb_rel[num_rel:] + self.drop(upd_rel)
        new_pred_emb_rel = upd_rel + self.drop(self.mlp_rel(upd_rel))

        return new_pred_emb_ent, new_pred_emb_rel

class KREPE(nn.Module):
    def __init__(self, act, dim, num_layer, num_head_ent, num_head_rel, model_dropout = 0.2, mask_eq_init = False):
        super(KREPE, self).__init__()
        layers = []

        for _ in range(num_layer):
            layers.append(KREPE_layer(act, dim, num_head_ent, num_head_rel, dropout = model_dropout))
        
        self.layers = nn.ModuleList(layers)

        self.dim = dim
        self.drop = nn.Dropout(p = model_dropout)
        self.mask_eq_init = mask_eq_init

        if not mask_eq_init:
            self.ent_init = nn.Parameter(torch.zeros(1, dim))
            self.rel_init = nn.Parameter(torch.zeros(1, dim))

        self.ent_mask = nn.Parameter(torch.zeros(1, dim))
        self.rel_mask = nn.Parameter(torch.zeros(1, dim))

        self.ent_norm = nn.LayerNorm(dim)
        self.rel_norm = nn.LayerNorm(dim)

        self.scale_ent = 1.0/math.sqrt(self.dim)
        self.scale_rel = 1.0/math.sqrt(self.dim)
        self.param_init()
    
    def param_init(self):
        if not self.mask_eq_init:
            nn.init.xavier_uniform_(self.ent_init)
            nn.init.xavier_uniform_(self.rel_init)
        nn.init.xavier_uniform_(self.ent_mask)
        nn.init.xavier_uniform_(self.rel_mask)

    def forward(self, pri, qual, qual2fact, num_ent, num_rel, \
                hpair, fact2hpair, \
                tpair, fact2tpair, \
                qpair, qual2qpair):

        device = pri.device

        hpair_ent = hpair[:, 0]
        hpair_rel = hpair[:, 1]

        tpair_ent = tpair[:, 0]
        tpair_rel = tpair[:, 1]

        qpair_ent = qpair[:, 1]
        qpair_rel = qpair[:, 0]
        
        if not self.mask_eq_init:
            emb_ent = self.ent_init.expand(num_ent, self.dim)
            emb_rel = self.rel_init.expand(num_rel, self.dim)
        else:
            emb_ent = self.ent_mask.expand(num_ent, self.dim)
            emb_rel = self.rel_mask.expand(num_rel, self.dim)

        emb_ents = [emb_ent]
        emb_rels = [emb_rel]

        for i, layer in enumerate(self.layers):
            if i % 1 == 0 and self.training:
                emb_ent, emb_rel = checkpoint(layer, emb_ent, emb_rel, \
                                                     pri, qual2fact, qual, \
                                                     hpair_ent, hpair_rel, fact2hpair, \
                                                     tpair_ent, tpair_rel, fact2tpair, \
                                                     qpair_ent, qpair_rel, qual2qpair, use_reentrant = False)
            else:
                emb_ent, emb_rel = layer(emb_ent, emb_rel, \
                                                pri, qual2fact, qual, \
                                                hpair_ent, hpair_rel, fact2hpair, \
                                                tpair_ent, tpair_rel, fact2tpair, \
                                                qpair_ent, qpair_rel, qual2qpair)
            emb_ents.append(emb_ent)
            emb_rels.append(emb_rel)

        return emb_ents, emb_rels
    
    def pred(self, pri, qual, qual2fact, \
             hpair, fact2hpair, \
             tpair, fact2tpair, \
             qpair, qual2qpair, \
             emb_ents, emb_rels, \
             num_ent_preds, num_rel_preds):
        #assumption: former indices are known entities, and latter indices are unknown entities

        device = emb_ents[0].device

        hpair_ent = hpair[:, 0]
        hpair_rel = hpair[:, 1]

        tpair_ent = tpair[:, 0]
        tpair_rel = tpair[:, 1]

        qpair_ent = qpair[:, 1]
        qpair_rel = qpair[:, 0]

        num_ent = len(emb_ents[0])
        num_rel = len(emb_rels[0])
        
        input_emb_ent = torch.cat([emb_ents[0], self.ent_mask.expand(num_ent_preds, self.dim)], dim = 0)
        input_emb_rel = torch.cat([emb_rels[0], self.rel_mask.expand(num_rel_preds, self.dim)], dim = 0)

        for emb_ent, emb_rel, layer in zip(emb_ents[1:], emb_rels[1:], self.layers):
            output_pred_emb_ent, output_pred_emb_rel = layer.pred(num_ent, num_rel, input_emb_ent, input_emb_rel,
                                                                  pri, qual2fact, qual,
                                                                  hpair_ent, hpair_rel, fact2hpair, 
                                                                  tpair_ent, tpair_rel, fact2tpair, 
                                                                  qpair_ent, qpair_rel, qual2qpair)

            input_emb_ent = torch.cat([emb_ent, output_pred_emb_ent], dim = 0)
            input_emb_rel = torch.cat([emb_rel, output_pred_emb_rel], dim = 0)

        ent_preds = torch.inner(self.ent_norm(output_pred_emb_ent), self.ent_norm(emb_ents[-1])) * self.scale_ent
        rel_preds = torch.inner(self.rel_norm(output_pred_emb_rel), self.rel_norm(emb_rels[-1])) * self.scale_rel
        
        return ent_preds, rel_preds