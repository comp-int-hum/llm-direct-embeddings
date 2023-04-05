from utility.corpus_utils import loadJsonCorpusDf, loadFCECorpusDf
import argparse
import pandas as pd
import os.path
import gzip
import numpy as np


def loadFullDf(corpus, corpus_name):

    if corpus_name == "mycorpus":
        df = loadJsonCorpusDf(gzip.open(corpus[0], "rt"))
    elif corpus_name == "fce-released-dataset":
        df = loadFCECorpusDf(corpus)

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", nargs="+", help="Path(dir or file)")
    parser.add_argument("--corpus_name", help="Namne of corpus")
    parser.add_argument("--outfile", dest="outfile",  help="Outfile of samples")


    args, rest = parser.parse_known_args()
    print(args.corpus)
    df = loadFullDf(args.corpus, args.corpus_name)
    #corpus_loc = os.path.join(args.corpus_dir, args.corpus)
    #if args.corpus == "mycorpus":
    #    df = loadJsonCorpusDf(gzip.open(os.path.join(corpus_loc,"corpus_0_1.gz"), "rt"))
    #elif args.corpus == "fce-released-dataset":
    #    df = loadFCECorpusDf(os.path.join(corpus_loc,"dataset/"))

    df.to_csv(args.outfile)