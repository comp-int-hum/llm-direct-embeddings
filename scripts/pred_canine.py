import argparse
import sys
import pandas as pd
import torch
import torch.nn.functional as F
from utility.corpus_utils import maskSample, loadBrownCorpusTree

from transformers import CanineModel
import jellyfish
import os
import logging

log_format = "%(asctime)s::%(filename)s::%(message)s"

logging.basicConfig(level='INFO', format=log_format)

#character embeddings as punishment matrix for ld?
#are the character embeddings of letters that can 

def get_hidden_states(encoded, char_span, model, layers):
    with torch.no_grad():
        output = model(encoded)

    states = output.hidden_states
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    char_encoded_output = output[char_span[0]:char_span[1]]
    return char_encoded_output


def pad_smaller_t(t1, t2):
	if t1.shape[0] < t2.shape[0]:
		return F.pad(t1, (0,0,0,t2.shape[0]-t1.shape[0])), t2
	elif t2.shape[0] < t1.shape[0]:
		return t1, F.pad(t2, (0,0,0,t1.shape[0]-t2.shape[0]))
	else:
		return t1, t2

def insert_alt_seq_at_index(orig, alt_seq, s_i, e_i):
	ins_str = orig[0:s_i] + alt_seq + orig[e_i:]
	new_final_index = s_i + len(alt_seq)
	encoded = torch.tensor([[ord(c) for c in ins_str]])
	return encoded, new_final_index


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("seglocs", help="CSV of segments")
	parser.add_argument("--maxlen", default=512,  help="Max unit (token) len for models")
	parser.add_argument("--pred_out", dest="outfile",  help="Outfile of predictions")
	parser.add_argument("--model", dest="model_name", help="A model name of a canine model")
	parser.add_argument("--max_ld", type=int, default=3,  help="Max LD")
	parser.add_argument("--layers", nargs="+", default=["-1"], dest="layers")
	parser.add_argument("--dictionary", default="brown")

	args, rest = parser.parse_known_args()

	s_df = pd.read_csv(args.seglocs)
	model = CanineModel.from_pretrained(args.model_name,output_hidden_states=True)
	layers = [int(l) for l in args.layers]
	bktree = loadBrownCorpusTree()

    with gzip.open(args.outfile, "wt") as of:
        for t,row in s_df.iterrows():
        	row_dict = {}

			s_i = row["i"]
			e_i = row["i"] + len(row["NS"])

			orig_str = row["sample"]
			orig_encoded = torch.tensor([[ord(c) for c in row["sample"]]])
			ground_inserted_encoded, g_i_e_i = insert_alt_seq_at_index(orig_str, row["Ground"], s_i, e_i)

			orig_embed = get_hidden_states(orig_encoded, [s_i,e_i], model, layers)
			ground_embed = get_hidden_states(ground_inserted_encoded, [s_i,g_i_e_i], model, layers)

			orig_mean = orig_embed.mean(dim=0)
			ground_mean = ground_embed.mean(dim=0)

			row_dict["ns_ground_cs"] = torch.cosine_similarity(orig_mean.reshape(1,-1), ground_mean.reshape(1,-1)).numpy().tolist()[0]

			orig_padded, ground_padded = pad_smaller_t(orig_embed, ground_embed)
			row_dict["ns_ground_cs_pad"] = torch.cosine_similarity(orig_padded.reshape(1,-1), ground_padded.reshape(1,-1)).numpy().tolist()[0]

			row_dict["ns_ground_ld"] = ns_ground_ld = jellyfish.levenshtein_distance(row["Ground"], row["NS"])

            logging.info("Sample: "+row["sample"])
            logging.info("Ground: "+row["Ground"])
            logging.info("NS: "+row["NS"])

			token_v_dict = {}
			for alt in bktree.find(row["NS"], args.max_ld):
				try:
					alt_inserted_encoded, new_ei = insert_alt_seq_at_index(orig_str, alt[1], s_i, e_i)
					alt_embed = get_hidden_states(alt_inserted_encoded, [s_i,new_ei], model, layers)

					alt_mean = alt_embed.mean(dim=0)
					cs_meaned_orig_alt = torch.cosine_similarity(orig_mean.reshape(1,-1), alt_mean.reshape(1,-1))

					orig_padded, alt_padded = pad_smaller_t(orig_embed, alt_embed)
					cs_padded_orig_alt = torch.cosine_similarity(orig_padded.reshape(1,-1), alt_padded.reshape(1,-1))

					token_v_dict[alt[1]] = {"LD":alt[0], "CS": cs_meaned_orig_alt.numpy().tolist()[0], "CS_Pad": cs_padded_orig_alt.numpy().tolist()[0]}

				except RuntimeError:
					logging.info("Error: Runtime")
					pass

            row_dict["alt_vec_dict"] = token_v_dict
            of.write(json.dumps(row_dict) + "\n")

"""
	def predCanine(row):
		s_i = row["i"]
		e_i = row["i"] + len(row["NS"])

		orig_str = row["sample"]
		orig_encoded = torch.tensor([[ord(c) for c in row["sample"]]])
		ground_inserted_encoded, g_i_e_i = insert_alt_seq_at_index(orig_str, row["Ground"], s_i, e_i)

		orig_embed = get_hidden_states(orig_encoded, [s_i,e_i], model, layers)
		ground_embed = get_hidden_states(ground_inserted_encoded, [s_i,g_i_e_i], model, layers)

		orig_mean = orig_embed.mean(dim=0)
		ground_mean = ground_embed.mean(dim=0)

		cs_meaned_orig_ground = torch.cosine_similarity(orig_mean.reshape(1,-1), ground_mean.reshape(1,-1)).numpy().tolist()[0]

		orig_padded, ground_padded = pad_smaller_t(orig_embed, ground_embed)
		cs_padded_orig_ground = torch.cosine_similarity(orig_padded.reshape(1,-1), ground_padded.reshape(1,-1)).numpy().tolist()[0]

		ns_ground_ld = jellyfish.levenshtein_distance(row["Ground"], row["NS"])
		alt_ids = [w for w in bktree.find(row["NS"], args.max_ld)] #maxld
		full_res_df = pd.DataFrame.from_records({"Alt":[a[1] for a in alt_ids], "LD":[a[0] for a in alt_ids]})

		logging.info(row["sample"])
		logging.info(row["Ground"])
		logging.info(row["NS"])

		def fullPreds(row):
			try:
				alt_inserted_encoded, new_ei = insert_alt_seq_at_index(orig_str, row["Alt"], s_i, e_i)
				alt_embed = get_hidden_states(alt_inserted_encoded, [s_i,new_ei], model, layers)

				alt_mean = alt_embed.mean(dim=0)
				cs_meaned_orig_alt = torch.cosine_similarity(orig_mean.reshape(1,-1), alt_mean.reshape(1,-1)).numpy().tolist()[0]

				orig_padded, alt_padded = pad_smaller_t(orig_embed, alt_embed)
				cs_padded_orig_alt = torch.cosine_similarity(orig_padded.reshape(1,-1), alt_padded.reshape(1,-1)).numpy().tolist()[0]
				return cs_meaned_orig_alt, cs_padded_orig_alt
			except RuntimeError:
				return 999,999


		full_res_df[["CSSimMean", "CSSimPad"]] = full_res_df.apply(fullPreds, axis=1, result_type="expand")
		os.path.join(os.path.dirname(args.outfile),str(row.name)+".csv")
		full_res_df.to_csv(os.path.join(os.path.dirname(args.outfile),str(row.name)+".csv"))

		min_ld = full_res_df[full_res_df.LD == full_res_df.LD.min()]
		max_cs_in_min_ld = min_ld[min_ld.CSSimMean == min_ld.CSSimMean.max()]
		max_cs_pad_in_min_ld = min_ld[min_ld.CSSimPad == min_ld.CSSimPad.min()]

		max_cs_mean = full_res_df[full_res_df.CSSimMean == full_res_df.CSSimMean.max()]
		max_cs_pad = full_res_df[full_res_df.CSSimPad == full_res_df.CSSimPad.max()]


		#print(max_cs_in_min_ld.Alt, max_cs_in_min_ld.CSSimMean)
		#print(max_cs_pad_in_min_ld.Alt, max_cs_pad_in_min_ld.CSSimPad)
		#print(cs_meaned_orig_ground, cs_padded_orig_ground, ns_ground_ld)
		logging.info(max_cs_in_min_ld.Alt.to_numpy()[0])
		logging.info(max_cs_in_min_ld.LD.to_numpy()[0])
		logging.info(max_cs_in_min_ld.CSSimMean.to_numpy()[0])
		logging.info(max_cs_pad_in_min_ld.Alt.to_numpy()[0])
		logging.info(max_cs_pad_in_min_ld.LD.to_numpy()[0])
		logging.info(max_cs_pad_in_min_ld.CSSimPad.to_numpy()[0])
		logging.info(max_cs_mean.Alt.to_numpy()[0])
		logging.info(max_cs_pad.Alt.to_numpy()[0])

		return cs_meaned_orig_ground, cs_padded_orig_ground, ns_ground_ld, max_cs_in_min_ld.Alt.to_numpy()[0], max_cs_in_min_ld.LD.to_numpy()[0], max_cs_in_min_ld.CSSimMean.to_numpy()[0], max_cs_pad_in_min_ld.Alt.to_numpy()[0], max_cs_pad_in_min_ld.LD.to_numpy()[0], max_cs_pad_in_min_ld.CSSimPad.to_numpy()[0], max_cs_mean.Alt.to_numpy()[0], max_cs_pad.Alt.to_numpy()[0]

	
	s_df[["NSGroundCSMean", "NSGroundCSPad", "NSGroundLD", "PredMean", "PredMeanLD", "PredMeanCS", "PredPad", "PredPadLD", "PredPadCS", "MaxCSMean", "MaxCSPad"]] = s_df.apply(predCanine, axis=1, result_type="expand")
	s_df.to_csv(args.outfile)
"""