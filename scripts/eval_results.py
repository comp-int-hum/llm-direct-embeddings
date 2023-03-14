import argparse
import pandas as pd
import gzip
import json


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("seg_results", nargs="+", help="Gzipped json files of segment distances")
	parser.add_argument("--char_level", action="store_true", help="If char level model, eval both mean and padded")
	parser.add_argument("--outfile")

	args, rest = parser.parse_known_args()

	print(args.seg_results)

	res = []
	for seg_result in args.seg_results:
		with gzip.open(seg_result,"rt") as inseg:
			for line in inseg:
				sample = json.loads(line)
				res_line = {"Sample":sample["Sample"], "NS":sample["NS"], "Ground":sample["Ground"]}
				fr_df = pd.DataFrame.from_dict(sample["alt_vec_dict"], orient="index")

				max_cs = fr_df[fr_df.CS == fr_df.CS.max()]
				res_line["Max_CS"] = max_cs.CS[0]
				res_line["Max_CS_Tok"] = max_cs.index[0]

				min_ld = fr_df[fr_df.LD == fr_df.LD.min()]
				max_cs_in_min_ld = min_ld[min_ld.CS == min_ld.CS.max()]
				res_line["Max_CS_Min_LD"] = max_cs_in_min_ld.CS[0]
				res_line["Max_CS_Min_LD_Tok"] = max_cs_in_min_ld.index[0]

				if args.char_level:
					max_cs_pad = max_cs = fr_df[fr_df.CS_Pad == fr_df.CS_Pad.max()]
					res_line["Max_CS_Pad"] = max_cs_pad.CS_Pad[0]
					res_line["Max_CS_Pad_Tok"] = max_cs_pad.index[0]

					max_cs_pad_in_min_ld = min_ld[min_ld.CS_Pad == min_ld.CS_Pad.max()]
					res_line["Max_CS_Pad_in_Min_LD"] = max_cs_pad_in_min_ld.CS_Pad[0]
					res_line["Max_CS_Pad_in_Min_LD_Tok"] = max_cs_pad_in_min_ld.index[0]
				
				res.append(res_line)

	pd.DataFrame.from_records(res).to_csv(args.outfile)