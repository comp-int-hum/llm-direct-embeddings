import logging
import json
import argparse
import gzip


import csv


log_format = "%(asctime)s::%(filename)s::%(message)s"

logging.basicConfig(level='INFO', format=log_format)


#combine embeddings? Train over known ground and created composites?

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("chunk_results", nargs="+", help="List of chunk results")
	parser.add_argument("--outfile", dest="outfile", help="Summary csv")
	parser.add_argument("--layers", nargs="+", dest="layers")
	parser.add_argument("--model_name", dest="model_name")


	args, rest = parser.parse_known_args()

	if args.model_name in ["google/canine-c", "general_character_bert","google/canine-s"]:
		args.layers = ["last"]

	with gzip.open(args.outfile,"wt",newline="") as csv_out:
		csvwriter = csv.writer(csv_out, delimiter=",")
		lsets = []
		for layer in args.layers:
			lsets += [layer+"_alt", layer+"_CD",layer+"_LD",layer+"_acc"]
		csvwriter.writerow(["observed","standard"] + lsets)
		for chunk in args.chunk_results:
			with gzip.open(chunk) as chunk_result:
				for s_result in chunk_result:
					s_result = json.loads(s_result)
					for annotation in s_result:
						base = [annotation["observed"], annotation["standard"]]
						for layer in args.layers:
							base.append(annotation["final_pred"][layer]["alt"])
							base.append(annotation["final_pred"][layer]["CD"])
							base.append(annotation["final_pred"][layer]["LD"])
							base.append(int(annotation["final_pred"][layer]["acc"]))
						csvwriter.writerow(base)

