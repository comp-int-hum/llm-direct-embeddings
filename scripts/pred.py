import torch
import logging
import json
import argparse
import gzip
import csv

from utility.pred_utils import LAYER_LOOKUP
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

log_format = "%(asctime)s::%(filename)s::%(message)s"

logging.basicConfig(level='INFO', format=log_format)

def checkWordEquality(standard, alt, alt_dict):
	if alt == standard:
		return True
	if standard in alt_dict:
		if alt_dict[standard] == alt:
			return True
	#if lemmatizer.lemmatize(alt) == lemmatizer.lemmatize(standard):
		#return True

	return False

def permissiveEquality(standard, alt, alt_dict):
	if checkWordEquality(standard, alt, alt_dict):
		return True
	if lemmatizer.lemmatize(alt) == lemmatizer.lemmatize(standard):
		return True
	if stemmer.stem(alt) == stemmer.stem(standard):
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

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("chunk_embed", help="Samples from chunk embedded")
	parser.add_argument("pred_out", help="Outfile of predictions")
	parser.add_argument("--model_name", dest="model_name", help="A model name of a bertlike model")
	parser.add_argument("--layers", nargs="+", dest="layers", help="Layers")
	parser.add_argument("--alternates_file", dest="alternates_file", help="A 2 column CSV file of alternate (say brit/us english) correct spellings")
	parser.add_argument("--max_ld", type=int, default=3, help="Max ld for prediction, up to 3")
	parser.add_argument("--ignore_an", type=bool, default=False)

	args, rest = parser.parse_known_args()

	conversion_dict = {}
	with open(args.alternates_file, "rt") as alts_in:
		alts_reader = csv.reader(alts_in)
		for line in alts_reader:
			conversion_dict[line[0]] = line[1]
		if args.ignore_an:
			conversion_dict["and"]="an'"
			


	if args.model_name in ["google/canine-c", "general_character_bert","google/canine-s"]:
		args.layers = ["last"]


	with gzip.open(args.chunk_embed, "rt") as embed_in, gzip.open(args.pred_out, "wt") as json_out:
		for e_l in embed_in:
			embeds = json.loads(e_l)
			total_n_annotations = len(embeds["annotations"]) + len(embeds["other_ided_ns"])

			out_l = []
			for annotation in embeds["annotations"]:
				final_pred = {l:{"alt":None, "CD": 0, "LD":1000, "acc":False, "acc_permissive":False, "std_cd": torch.cosine_similarity(torch.Tensor(annotation["standard_embeddings"][l]).mean(dim=0), torch.Tensor(annotation["observed_embeddings"][l]).mean(dim=0), dim=0).item()} for l in args.layers}
				final_pred_only_model = {l:{"alt":None, "CD": 0, "LD":1000, "acc":False, "acc_permissive":False} for l in args.layers}
				alt_preds = {}
				standard_in_alts = checkStandardInAlts(annotation["standard"], annotation["alts"], conversion_dict)
				for alt, embed in annotation["alts"].items():
					if embed["LD"] <= args.max_ld:
						alt_preds[alt] = {l:{} for l in args.layers}

						for layer in args.layers:
						#meaning pieces "under investigation"
							pred = {"LD":embed["LD"], "CD":torch.cosine_similarity(torch.Tensor(annotation["observed_embeddings"][layer]).mean(dim=0), torch.Tensor(embed["embed"][layer]).mean(dim=0), dim=0).item()}
							alt_preds[alt][layer] = pred
							if pred["CD"] > final_pred[layer]["CD"] and pred["LD"] <= final_pred[layer]["LD"]:
								final_pred[layer] = pred
								final_pred[layer]["alt"] = alt
								final_pred[layer]["acc"] = checkWordEquality(annotation["standard"], alt, conversion_dict)
								final_pred[layer]["acc_permissive"] = permissiveEquality(annotation["standard"], alt, conversion_dict)
							if pred["CD"] > final_pred_only_model[layer]["CD"] or ((pred["CD"] == final_pred_only_model[layer]["CD"]) and (pred["LD"] <= final_pred_only_model[layer]["LD"])):
								final_pred_only_model[layer] = pred
								final_pred_only_model[layer]["alt"] = alt
								final_pred_only_model[layer]["acc"] = checkWordEquality(annotation["standard"], alt, conversion_dict)
								final_pred_only_model[layer]["acc_permissive"] = permissiveEquality(annotation["standard"], alt, conversion_dict)

				out_l.append({"preds": alt_preds, "final_pred":final_pred, "final_pred_model_only":final_pred_only_model, "observed":annotation["observed"], "standard": annotation["standard"], "alt_present":int(standard_in_alts), "num_alts":len(annotation["alts"]), "total_ns_in_sample":total_n_annotations})


			json_out.write(json.dumps(out_l) + "\n")


