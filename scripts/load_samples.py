from utility.corpus_utils import loadJsonCorpusDf, loadFCECorpusDf
import argparse
import pandas as pd
import os.path
import gzip
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Corpus name (direct dir)")
    parser.add_argument("--corpus_dir", default = "", help="Bottom level corpus dir")
    parser.add_argument("--outfiles", nargs="+", dest="outfile",  help="Outfile of samples")
    parser.add_argument("--chunks", default=1, help="Number of chunks for sample doc (for SCONS paralellization)")


    args, rest = parser.parse_known_args()
    corpus_loc = os.path.join(args.corpus_dir, args.corpus)

    if args.corpus == "mycorpus":
        df = loadJsonCorpusDf(gzip.open(os.path.join(corpus_loc,"corpus_0_1.gz"), "rt"))
    elif args.corpus == "fce-released-dataset":
        df = loadFCECorpusDf(os.path.join(corpus_loc,"dataset/"))

    split = np.array_split(df, int(args.chunks))
    for chunk,name in zip(split, args.outfile):
        chunk.to_csv(name)