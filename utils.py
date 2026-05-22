import numpy as np
import torch

def calculate_ranks(score, targets, filter_list):
	device = score.device
	cmp_targets = torch.ones(len(score), dtype = torch.bool, device = device)
	cmp_targets[filter_list] = False
	filtered_scores = score[cmp_targets].unsqueeze(dim = 0)

	score_targets = score[targets].unsqueeze(dim = 1)
	
	ranks = (filtered_scores > score_targets).sum(dim = 1) + (filtered_scores == score_targets).sum(dim = 1) // 2 + 1

	return ranks

def metrics(rank):
    mr = np.mean(rank)
    mrr = np.mean(1 / rank)
    hit10 = np.sum(rank < 11) / len(rank)
    hit3 = np.sum(rank < 4) / len(rank)
    hit1 = np.sum(rank < 2) / len(rank)
    return mr, mrr, hit10, hit3, hit1