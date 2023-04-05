import argparse
import sys

from transformers import BertTokenizer
from modeling.character_bert import CharacterBertModel
from utils.character_cnn import CharacterIndexer
from utility.corpus_utils import maskSample

from utility.pred_utils import LAYER_LOOKUP

import torch
import os

import logging
import json

log_format = "%(asctime)s::%(filename)s::%(message)s"

logging.basicConfig(level='INFO', format=log_format)



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
	return embeddings_for_batch[tok_range]

def insert_over_mask_token_at_index(tokenized, index, new_token):
	return tokenized[0:index] + [new_token] + tokenized[index+3:]


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("chunk_json", nargs="+", help="CSV of segments")
	parser.add_argument("--maxlen", default=512,  help="Max unit (token) len for models")
	parser.add_argument("--embeddings_out", nargs="+", dest="outfiles",  help="Outfile of predictions")

	args, rest = parser.parse_known_args()


	b_t = BertTokenizer.from_pretrained("bert-base-uncased")
	indexer = CharacterIndexer()
	model = CharacterBertModel.from_pretrained(
	"./pretrained-models/general_character_bert/",output_hidden_states=True)



	for in_samp,out_embed in zip(args.chunk_json, args.outfiles):
		with open(in_samp, "r") as j_in:
			json_sample = json.load(j_in) 


		masked_sample = maskSample(json_sample) #going to use mask as a place to insert candidates



		x = b_t.basic_tokenizer.tokenize(masked_sample)
		sli = find_sublist(["[","mask","]"], x)
		if len(sli) != 1:
			logging.info("mask not one")
			logging.info(x)
			pass

		tok_index = sli[0][0]+1
		x = ["[CLS]", *x, "[SEP]"]
		x2 = insert_over_mask_token_at_index(x, tok_index, json_sample["NS"])

		NS_e = get_embedding_cbert(model,x2,tok_index,indexer)

		x2 = insert_over_mask_token_at_index(x, tok_index, json_sample["Ground"])
		ground_vec = get_embedding_cbert(model,x2,tok_index,indexer)



		logging.info(json_sample["sample"])
		logging.info(json_sample["Ground"])
		logging.info(json_sample["NS"])

		out_d = {json_sample["NS"]: NS_e, json_sample["Ground"]: ground_vec}
		print(out_d)
		for alt in json_sample["Alts"]:
			x2 = insert_over_mask_token_at_index(x,tok_index,alt)
			alt_e = {"last": get_embedding_cbert(model, x2, tok_index, indexer)}
			out_d[alt] = alt_e

		torch.save(out_d,out_embed)