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
	parser.add_argument("--ld", default=3, help="max ld of the run")


	args, rest = parser.parse_known_args()


	with open(args.accurate,"wt",newline="") as accurate_out, open(args.inaccurate,"wt",newline="") as inaccurate_out:
		accurate_writer = csv.writer(accurate_out, delimiter=",")
		inaccurate_writer = csv.writer(inaccurate_out, delimiter=",")
		accurate_writer.writerow(["observed", "standard", "alt", "CD", "LD", "acc", "alt_present", "total_ns_in_sample", "acc_permissive"])
		inaccurate_writer.writerow(["observed", "standard", "alt", "CD", "LD", "acc", "alt_present", "total_ns_in_sample", "acc_permissive", "acc_alt_ld", "standard_rank", "standard_rank_present_only"])

		running_mrr = 0
		running_mrr_in_alts = 0
		running_mrr_mo = 0
		running_mrr_mo_in_alts = 0
		sample_total = 0
		in_alts_total = 0
		correct_total = 0
		inaccurate_not_in_alts = 0
		inaccurate = 0
		only_ns_total = 0
		only_ns_accurate = 0
		permissive_acc = 0
		acc_model_only = 0
		acc_permissive_model_only = 0
		acc_permissive_single_model_only = 0
		for chunk in args.chunk_results:
			with gzip.open(chunk) as chunk_result:
				for s_result in chunk_result:
					s_result = json.loads(s_result)
					for annotation in s_result:
						sample_total += 1
						running_mrr += (1/annotation["std_ranks"][args.layers]) if annotation["std_ranks"][args.layers] != 0 else 0
						running_mrr_mo += (1/annotation["mo_std_ranks"][args.layers]) if annotation["mo_std_ranks"][args.layers] != 0 else 0
						running_mrr_mo_in_alts += (1/annotation["mo_std_ranks"][args.layers]) if (annotation["mo_std_ranks"][args.layers] != 0 and annotation["alt_present"] == 1) else 0
						running_mrr_in_alts += (1/annotation["std_ranks"][args.layers]) if (annotation["std_ranks"][args.layers] != 0 and annotation["alt_present"] == 1) else 0
						if int(annotation["total_ns_in_sample"]) == 1:
							only_ns_total += 1
						in_alts_total += annotation["alt_present"]
						row = [annotation["observed"], annotation["standard"], annotation["final_pred"][args.layers]["alt"], 
							annotation["final_pred_model_only"][args.layers]["CD"], annotation["final_pred_model_only"][args.layers]["LD"], int(annotation["final_pred_model_only"][args.layers]["acc"]), annotation["alt_present"], annotation["total_ns_in_sample"], annotation["final_pred_model_only"][args.layers]["acc_permissive"]]

						if annotation["final_pred_model_only"][args.layers]["acc_permissive"]:
							acc_permissive_model_only+=1
						if annotation["final_pred"][args.layers]["acc_permissive"]:
							permissive_acc += 1
							if int(annotation["total_ns_in_sample"]) == 1:
								acc_permissive_single_model_only += 1
						if annotation["final_pred_model_only"][args.layers]["acc"]:
							acc_model_only += 1

						if annotation["final_pred"][args.layers]["acc"]:
							correct_total += 1
							if int(annotation["total_ns_in_sample"]) == 1:
								only_ns_accurate += 1
							accurate_writer.writerow(row)
						else:
							if annotation["alt_present"] == 0:
								inaccurate_not_in_alts += 1
							else:
								try:
									row.append(annotation["preds"][annotation["standard"]][args.layers]["LD"])
								except KeyError:
									pass
							row.append(annotation["std_ranks"][args.layers])
							inaccurate_writer.writerow(row)

		n_inaccurate = sample_total-correct_total
		accuracy = float(correct_total)/sample_total
		observed_in_alts_per = float(in_alts_total)/sample_total
		accuracy_alt_present = float(correct_total)/(sample_total-inaccurate_not_in_alts)
		stem_acc = float(permissive_acc)/sample_total
		stem_acc_alt_present = float(permissive_acc)/(sample_total-inaccurate_not_in_alts)
		with open(args.summary,"wt", newline="") as summary_out:
			sw = csv.writer(summary_out)
			sw.writerow(["Model", "Max LD", "N", "n_Acc", "n_Inacc", "Accuracy", "MRR", "MRR_standard_present", "In alts %", "Inacc not in alts", "Acc alt present", "Acc 1ns", "N 1ns", "Stemmed acc", "Stemmed acc alt present", "Acc only model", "MRR model only", "MRR model only acc present"])
			sw.writerow([args.summary, args.ld, sample_total, (sample_total - n_inaccurate), n_inaccurate,
				     accuracy,
				     running_mrr/sample_total,
				     running_mrr_in_alts/in_alts_total,
				     observed_in_alts_per,
				     inaccurate_not_in_alts,
				     accuracy_alt_present,float(only_ns_accurate)/only_ns_total,
				     only_ns_total, stem_acc,
				     stem_acc_alt_present,
				     float(acc_model_only)/sample_total,
				     running_mrr_mo/sample_total,
				     running_mrr_mo_in_alts/in_alts_total])
			
