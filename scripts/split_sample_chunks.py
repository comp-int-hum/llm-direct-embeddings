import argparse
import pandas as pd
import os.path
import numpy as np
from utility.corpus_utils import loadBrownCorpusTree
import json


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Full CSV corpus")
    parser.add_argument("outfiles", nargs="+",  help="Outfile names for chunk")
    parser.add_argument("--chunk_indices", nargs="+")
    parser.add_argument("--max_ld", default=3)

    args, rest = parser.parse_known_args()

    bktree = loadBrownCorpusTree()

    c_df = pd.read_csv(args.corpus, index_col=0)
    for t,of in zip(c_df.iloc[args.chunk_indices].itertuples(), args.outfiles):

    	js = json.loads(pd.DataFrame([t]).to_json(force_ascii=False, orient="index"))["0"]
    	alts = bktree.find(js["NS"], args.max_ld)
    	js["Alts"] = {a[1]: a[0] for a in alts}
    	with open(of,"w") as out:
    		json.dump(js, out)
