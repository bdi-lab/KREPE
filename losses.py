import torch.nn.functional as F
import torch

def kDCE_loss(k_per_fact, preds, pred_idxs, answers, fact_lengths):
    B = k_per_fact.shape[0] 
    device = preds.device
    dtype = preds.dtype
    token_losses = F.cross_entropy(preds, answers, reduction = "none")
    loss_per_fact = torch.zeros(B, device=device, dtype=dtype).index_add(dim=0, index=pred_idxs, source=token_losses)
    final_loss_per_fact = (loss_per_fact / k_per_fact) * fact_lengths
    return final_loss_per_fact.mean()
