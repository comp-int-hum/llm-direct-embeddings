import argparse
import sys
import pandas as pd
import torch

import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="Results CSV")
    parser.add_argument("--eval_out", help="Evaluation outfile")


    args, rest = parser.parse_known_args()
    corpus_loc = os.path.join(args.corpus_dir, args.corpus)

    print(args.results_file, args.eval_out)
    #results = pd.read_csv(results_file)

    #print(results)
