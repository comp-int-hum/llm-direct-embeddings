import torch
import jellyfish
import logging
import json
import argparse

from utility.pred_utils import LAYER_LOOKUP


log_format = "%(asctime)s::%(filename)s::%(message)s"

logging.basicConfig(level='INFO', format=log_format)


#combine embeddings? Train over known ground and created composites?

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("chunk_embeds", nargs="+", help="Collection of embed files")
	parser.add_argument("--pred_out", dest="outfile", nargs="+",  help="Outfile of predictions")
	parser.add_argument("--chunk", nargs="+")
	parser.add_argument("--model_name", dest="model_name", help="A model name of a bertlike model")
	parser.add_argument("--layers", nargs="+", dest="layers", help="Layers")


	args, rest = parser.parse_known_args()


	if args.model_name in ["google/canine-c", "general_character_bert","google/canine-s"]:
		args.layers = ["last"]

	for embeds,sample,outfile in zip(args.chunk_embeds, args.chunk, args.outfile):
		embeds = torch.load(embeds)
		with open(sample, "r") as s_in:
			sample = json.load(s_in)

		ns_ground_ld = jellyfish.levenshtein_distance(sample["Ground"], sample["NS"])

		out_d = {"NS_Ground_LD": ns_ground_ld, "Layers":{l:{} for l in args.layers}}

		for layer in args.layers:

			#meaning pieces "under investigation"
			ns_ground_cd = torch.cosine_similarity(embeds[sample["Ground"]][layer].mean(dim=0).reshape(1,-1), embeds[sample["NS"]][layer].mean(dim=0).reshape(1,-1)).numpy().tolist()[0]

			layer_l = []

			final_prediction = {"Alt":None, "CD": 0, "LD":1000}

			for alt, LD in sample["Alts"].items():
				pred = {"Alt":alt, "LD":LD, "CD":torch.cosine_similarity(embeds[sample["NS"]][layer].mean(dim=0).reshape(1,-1), embeds[alt][layer].mean(dim=0).reshape(1,-1)).numpy().tolist()[0]}
				layer_l.append(pred)
				if pred["CD"] > final_prediction["CD"] and pred["LD"] <= final_prediction["LD"]:
					final_prediction = pred

			acc = True if final_prediction["Alt"].lower().strip() == sample["Ground"].lower().strip() else False

			out_d["Layers"][layer] = {"Preds": layer_l, "Final_Pred":final_prediction, "NS_Ground_CD": ns_ground_cd, "Acc":acc}

			logging.info("Layer: %s", layer)
			logging.info("Ground: %s", sample["Ground"])
			logging.info("Pred: %s", final_prediction["Alt"])
			logging.info("PredCD: %f", final_prediction["CD"])
			logging.info("PredLD: %f", final_prediction["LD"])
			logging.info("Acc %s" str(acc))
		with open(outfile,"w") as p_out:
			json.dump(out_d,p_out)


