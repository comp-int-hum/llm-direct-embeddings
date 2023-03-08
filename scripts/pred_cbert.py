import argparse
import sys

from transformers import BertTokenizer
from modeling.character_bert import CharacterBertModel
from utils.character_cnn import CharacterIndexer
from utility.corpus_utils import maskSample, loadBrownCorpusTree
from utility.pred_utils import fetchMaxCSandMinEucinMaxLD

import torch
import os

import pandas as pd
import jellyfish


def find_sublist(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

def get_embedding_cbert(model, x, tok_range, indexer):
	x_ids = indexer.as_padded_tensor(x)
	with torch.no_grad():
		embeddings_for_batch, _ = model(x_ids)
	return torch.nan_to_num(embeddings_for_batch[tok_range])

def insert_over_mask_token_at_index(tokenized, index, new_token):
	return tokenized[0:index] + [new_token] + tokenized[index+3:]


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("seglocs", help="CSV of segments")
	parser.add_argument("--maxlen", default=512,  help="Max unit (token) len for models")
	parser.add_argument("--pred_out", dest="outfile",  help="Outfile of predictions")
	parser.add_argument("--max_ld", type=int, default=3,  help="Max LD")
	parser.add_argument("--dictionary", default="brown")

	args, rest = parser.parse_known_args()

	s_df = pd.read_csv(args.seglocs)

	b_t = BertTokenizer.from_pretrained("bert-base-uncased")
	indexer = CharacterIndexer()
	model = CharacterBertModel.from_pretrained(
	"./pretrained-models/general_character_bert/",output_hidden_states=True)

	bktree = loadBrownCorpusTree()

	s_df["MaskedSample"] = s_df.apply(maskSample, mask_token=b_t.mask_token, axis=1) #going to use mask as a place to insert candidates

	def predCBERT(row):
		x = b_t.basic_tokenizer.tokenize(row["MaskedSample"])
		sli = find_sublist(["[","mask","]"], x)
		if len(sli) != 1:
			print("mask not one")
			print(x)
			return 0,0,0,0,0,0,0,0,0

		tok_index = sli[0][0]+1
		x = ["[CLS]", *x, "[SEP]"]
		x2 = insert_over_mask_token_at_index(x, tok_index, row["NS"])
		#x2 = x[0:tok_index] + [row["NS"]] + x[tok_index+3:]
		NS_e = get_embedding_cbert(model,x2,tok_index,indexer)

		x2 = insert_over_mask_token_at_index(x, tok_index, row["Ground"])
		ground_vec = get_embedding_cbert(model,x2,tok_index,indexer)

		ns_ground_cs = torch.cosine_similarity(NS_e.reshape(1,-1), ground_vec.reshape(1,-1)).numpy().tolist()[0]
		ns_ground_ld = jellyfish.levenshtein_distance(row["Ground"], row["NS"])

		alt_ids = [w for w in bktree.find(row["NS"], args.max_ld)] #maxld
		full_res_df = pd.DataFrame.from_records({"Alt":[a[1] for a in alt_ids], "LD":[a[0] for a in alt_ids]})

		def fullPreds(row):
			x2 = insert_over_mask_token_at_index(x, tok_index, row["Alt"])
			alt_e = get_embedding_cbert(model, x2, tok_index, indexer)
			return torch.cosine_similarity(alt_e.reshape(1,-1), NS_e.reshape(1,-1)).numpy().tolist()[0], torch.cdist(alt_e.reshape(1,-1), NS_e.reshape(1,-1)).numpy().tolist()[0][0]

		full_res_df[["CSSim","EucDist"]] = full_res_df.apply(fullPreds, axis=1, result_type="expand")
		full_res_df.to_csv(os.path.join(os.path.dirname(args.outfile),str(row.name)+".csv"))
		
		max_cs_in_min_ld, min_euc_in_min_ld = fetchMaxCSandMinEucinMaxLD(full_res_df)
		print(max_cs_in_min_ld.Alt, max_cs_in_min_ld.CSSim)


		max_cs = full_res_df[full_res_df.CSSim == full_res_df.CSSim.max()]
		return ns_ground_cs, ns_ground_ld, max_cs_in_min_ld.Alt, max_cs_in_min_ld.LD, max_cs_in_min_ld.CSSim, min_euc_in_min_ld.Alt, min_euc_in_min_ld.LD, min_euc_in_min_ld.EucDist, max_cs.Alt



	s_df[["NS_GroundCS", "NS_GroundLD", "Pred", "PredLD","PredCS", "PredEuc", "PredEucLD", "PredEucEuc", "Max_CSAlt"]] = s_df.apply(predCBERT, axis=1, result_type="expand")
	s_df.to_csv(args.outfile)