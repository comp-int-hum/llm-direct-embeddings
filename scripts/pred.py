import torch
import logging
import json
import argparse
import gzip

from utility.pred_utils import LAYER_LOOKUP


log_format = "%(asctime)s::%(filename)s::%(message)s"

logging.basicConfig(level='INFO', format=log_format)


#combine embeddings? Train over known ground and created composites?

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("chunk_embed", help="Samples from chunk embedded")
	parser.add_argument("pred_out", help="Outfile of predictions")
	parser.add_argument("--model_name", dest="model_name", help="A model name of a bertlike model")
	parser.add_argument("--layers", nargs="+", dest="layers", help="Layers")


	args, rest = parser.parse_known_args()


	if args.model_name in ["google/canine-c", "general_character_bert","google/canine-s"]:
		args.layers = ["last"]


	with gzip.open(args.chunk_embed, "rt") as embed_in, gzip.open(args.pred_out, "wt") as json_out:
		for e_l in embed_in:
			embeds = json.loads(e_l)

			out_l = []
			for annotation in embeds["embeds"]:
				final_pred = {l:{"alt":None, "CD": 0, "LD":1000, "acc":False} for l in args.layers}
				alt_preds = {}
				for alt in annotation["alts"]:
					alt_preds[alt["token"]] = {l:{} for l in args.layers}
					for layer in args.layers:
						#meaning pieces "under investigation"
						
						pred = {"LD":alt["LD"], "CD":torch.cosine_similarity(torch.Tensor(annotation["observed"]["embed"][layer]).mean(dim=0).reshape(1,-1), torch.Tensor(alt["embed"][layer]).mean(dim=0).reshape(1,-1)).numpy().tolist()[0]}
						alt_preds[alt["token"]][layer] = pred
						if pred["CD"] > final_pred[layer]["CD"] and pred["LD"] <= final_pred[layer]["LD"]:
							final_pred[layer] = pred
							final_pred[layer]["alt"] = alt["token"]
							final_pred[layer]["acc"] = True if alt["token"].strip() == annotation["standard"]["token"].strip() else False

				out_l.append({"preds": alt_preds, "final_pred":final_pred, "observed":annotation["observed"]["token"], "standard": annotation["standard"]["token"]})


			json_out.write(json.dumps(out_l) + "\n")


