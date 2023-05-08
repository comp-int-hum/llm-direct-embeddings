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
	parser.add_argument("--accurate", dest="accurate", help="Accurate results")
	parser.add_argument("--inaccurate", dest="inaccurate", help="Inaccurate results")
	parser.add_argument("--summary", dest="summary", help="Summary text file")
	parser.add_argument("--layers", dest="layers")


	args, rest = parser.parse_known_args()

	with open(args.accurate,"wt",newline="") as accurate_out, open(args.inaccurate,"wt",newline="") as inaccurate_out:
		accurate_writer = csv.writer(accurate_out, delimiter=",")
		inaccurate_writer = csv.writer(inaccurate_out, delimiter=",")
		accurate_writer.writerow(["observed", "standard", "alt", "CD", "LD", "acc", "alt_present"])
		inaccurate_writer.writerow(["observed", "standard", "alt", "CD", "LD", "acc", "alt_present"])

		sample_total = 0
		in_alts_total = 0
		correct_total = 0
		inaccurate_not_in_alts = 0
		for chunk in args.chunk_results:
			with gzip.open(chunk) as chunk_result:
				for s_result in chunk_result:
					s_result = json.loads(s_result)
					for annotation in s_result:
						sample_total += 1
						in_alts_total += annotation["alt_present"]
						row = [annotation["observed"], annotation["standard"], annotation["final_pred"][args.layers]["alt"], 
							annotation["final_pred"][args.layers]["CD"], annotation["final_pred"][args.layers]["LD"], int(annotation["final_pred"][args.layers]["acc"]), annotation["alt_present"]]

						if annotation["final_pred"][args.layers]["acc"]:
							correct_total += 1
							accurate_writer.writerow(row)
						else:
							inaccurate_writer.writerow(row)
							if annotation["alt_present"] == 0:
								inaccurate_not_in_alts += 1

		with open(args.summary,"wt") as summary_out:
			summary_out.write("Num accurate: " + str(correct_total) + "\n")
			summary_out.write("Num inaccurate: " + str(sample_total-correct_total) + "\n")
			summary_out.write("Accuracy: " + str(float(correct_total)/sample_total) + "\n")
			summary_out.write("Observed in alts percentage: " + str(float(in_alts_total)/sample_total) + "\n")
			summary_out.write("Inaccurate and not observed in alts: " + str(inaccurate_not_in_alts) + "\n")
		

		"""
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

		"""