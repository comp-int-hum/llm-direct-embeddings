import torch
import logging
import json
import argparse
import gzip

from utility.pred_utils import LAYER_LOOKUP
from nltk.stem import WordNetLemmatizer

log_format = "%(asctime)s::%(filename)s::%(message)s"

logging.basicConfig(level='INFO', format=log_format)

def checkWordEquality(standard, alt, alt_dict):
	if alt == standard:
		return True
	if lemmatizer.lemmatize(alt) == lemmatizer.lemmatize(standard):
		return True
	if standard in alt_dict:
		if standard == alt_dict[standard]:
			return True

	return False

def checkStandardInAlts(standard, alts, alt_dict):
	if standard in alts:
		return True
	if standard in alt_dict:
		if alt_dict[standard] in alts:
			return True
	return False


#combine embeddings? Train over known ground and created composites?

lemmatizer = WordNetLemmatizer()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("chunk_embed", help="Samples from chunk embedded")
	parser.add_argument("pred_out", help="Outfile of predictions")
	parser.add_argument("--model_name", dest="model_name", help="A model name of a bertlike model")
	parser.add_argument("--layers", nargs="+", dest="layers", help="Layers")
	parser.add_argument("--alternates_file", dest="alternates_file", help="A 2 column CSV file of alternate (say brit/us english) correct spellings")


	args, rest = parser.parse_known_args()

	conversion_dict = {}
	with open(args.alternates_file, "rt") as alts_in:
		for line in alts_in:
			split_line = line.split(",")
			conversion_dict[split_line[0].strip()] = split_line[1].strip()




	if args.model_name in ["google/canine-c", "general_character_bert","google/canine-s"]:
		args.layers = ["last"]


	with gzip.open(args.chunk_embed, "rt") as embed_in, gzip.open(args.pred_out, "wt") as json_out:
		for e_l in embed_in:
			embeds = json.loads(e_l)
			total_n_annotations = len(embeds["annotations"]) + len(embeds["other_ided_ns"])

			out_l = []
			for annotation in embeds["annotations"]:
				final_pred = {l:{"alt":None, "CD": 0, "LD":1000, "acc":False} for l in args.layers}
				alt_preds = {}
				for alt, embed in annotation["alts"].items():
					alt_preds[alt] = {l:{} for l in args.layers}

					for layer in args.layers:
						#meaning pieces "under investigation"
						pred = {"LD":embed["LD"], "CD":torch.cosine_similarity(torch.Tensor(annotation["observed_embeddings"][layer]).mean(dim=0).reshape(1,-1), torch.Tensor(embed["embed"][layer]).mean(dim=0).reshape(1,-1)).numpy().tolist()[0]}
						alt_preds[alt][layer] = pred
						if pred["CD"] > final_pred[layer]["CD"] and pred["LD"] <= final_pred[layer]["LD"]:
							final_pred[layer] = pred
							final_pred[layer]["alt"] = alt
							final_pred[layer]["acc"] = checkWordEquality(annotation["standard"], alt, conversion_dict)

				out_l.append({"preds": alt_preds, "final_pred":final_pred, "observed":annotation["observed"], "standard": annotation["standard"], "alt_present":int(checkStandardInAlts(annotation["standard"], annotation["alts"], conversion_dict)), "num_alts":len(annotation["alts"]), "total_ns_in_sample":total_n_annotations})


			json_out.write(json.dumps(out_l) + "\n")


