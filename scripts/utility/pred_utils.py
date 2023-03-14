import torch
import pandas as pd
import torch.nn.functional as F


def fetchMaxCSandMinEucinMaxLD(full_res_df):
	min_ld = full_res_df[full_res_df.LD == full_res_df.LD.min()]
	max_cs_in_min_ld = min_ld[min_ld.CSSim == min_ld.CSSim.max()]
	min_euc_in_min_ld = min_ld[min_ld.EucDist == min_ld.EucDist.min()]

	return max_cs_in_min_ld, min_euc_in_min_ld


def pad_smaller_t(t1, t2):
	if t1.shape[0] < t2.shape[0]:
		return F.pad(t1, (0,0,0,t2.shape[0]-t1.shape[0])), t2
	elif t2.shape[0] < t1.shape[0]:
		return t1, F.pad(t2, (0,0,0,t1.shape[0]-t2.shape[0]))
	else:
		return t1, t2