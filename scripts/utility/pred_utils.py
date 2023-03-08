import torch
import pandas as pd



def fetchMaxCSandMinEucinMaxLD(full_res_df):
	min_ld = full_res_df[full_res_df.LD == full_res_df.LD.min()]
	max_cs_in_min_ld = min_ld[min_ld.CSSim == min_ld.CSSim.max()]
	min_euc_in_min_ld = min_ld[min_ld.EucDist == min_ld.EucDist.min()]

	return max_cs_in_min_ld, min_euc_in_min_ld
