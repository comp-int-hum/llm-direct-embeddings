import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
import steamroller
import glob
import pandas as pd
from itertools import islice
import numpy as np
from collections import defaultdict


from scripts.utility.corpus_utils import loadFCECorpusDf, loadJsonCorpusDf

# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle


#split maximally (into samples), generate list of all, then rejoin into chunks in a chunk step, so then i will have a list of these individual sample files resident in each chunk.
#can then produce layer and model outputs for embeddings over each chunk, and pass that into model retrieval (so open appropriate chunk, do stuff, save to paired output chunk)


vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 5000),
    ("MODELS","",["google/canine-c", "bert-large-uncased", "roberta-large"]),
    #("MODELS","",["bert-large-uncased", "google/canine-c", "roberta-large", "general_character_bert"]),
    #("MODELS", "", ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large", "general_character_bert", "google/canine-c", "google/canine-s"]),
    #("LAYERS","", [[-1,-2,-3,-4], [-1], [1], [2], [3], [1,2,3], [6]]),
    ("LAYERS","",["last","last_four", "first_three", "middle"]),
    ("DATASETS", "", [["fce-released-dataset", 3670], ["mycorpus",1324]]), #,"fce-released-dataset"]),
    ("CORPORA_DIR","","corpora"),
    ("RANDOM_STATE","", 10),
    ("NUM_CHUNKS","",5),
    ("MAX_LD","",3)
)


env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", steamroller.generate],
)

env.AddBuilder(
    "LoadSamples",
    "scripts/load_samples.py",
    "${SOURCES} --corpus_name ${CORPUS_NAME} --outfile ${TARGETS[0]}",
)

env.AddBuilder(
    "SplitToChunks",
    "scripts/split_sample_chunks.py",
    "${SOURCES[0]} ${TARGETS} --chunk_indices ${CHUNK_INDICES}"

    )

env.AddBuilder(
    "EmbedBertlike",
    "scripts/get_tensors_bertlike.py",
    "${SOURCES} --embeddings_out ${TARGETS} --model ${MODEL_NAME} --layers ${LAYERS}"
)


env.AddBuilder(
    "EmbedCanine",
    "scripts/get_tensors_canine.py",
    "${SOURCES} --model ${MODEL_NAME} --embeddings_out ${TARGETS} --layers ${LAYERS}"
    )

env.AddBuilder(
    "EmbedCBert",
    "scripts/get_tensors_cbert.py",
    "${SOURCES} --model ${MODEL_NAME} --embeddings_out ${TARGETS}"
    )

env.AddBuilder(
    "Pred",
    "scripts/pred.py",
    "${SOURCES} --pred_out ${TARGETS} --chunk ${CHUNK} --model_name ${MODEL_NAME} --layers ${LAYERS}"

    )




env.AddBuilder(
    "EvalResults",
    "scripts/eval_results.py",
    "${SOURCES} --outfile ${TARGETS[0]}"
)


# function for width-aware printing of commands
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print(s[:int(float(env["OUTPUT_WIDTH"]) / 2) - 2] + "..." + s[-int(float(env["OUTPUT_WIDTH"]) / 2) + 1:])
    else:
        print(s)


env['PRINT_CMD_LINE_FUNC'] = print_cmd_line


env.Decider("timestamp-newer")



#blowing up the depenedency graph -- going to have to go back to chunk files rather than logically collected chunks of individual files
#still good to have pushed the layers stuff out
#should simplify loading
#so chunk script, with files = nchunks as out, corpus as in
#iterate chunks



#chunk into X pieces for use with -j --jobs (so can implicitly multicore over x processors)



for dataset_name in env["DATASETS"]:
    if dataset_name[0] == "fce-released-dataset":
        d_samples_full = env.LoadSamples(os.path.join("work","fce-released-dataset","full.csv"),[f for f in glob.glob(env["CORPORA_DIR"]+"/fce-released-dataset"+"/dataset"+"/*/*.xml", recursive = True)], CORPUS_NAME=dataset_name[0])
    elif dataset_name[0] == "mycorpus":
        d_samples_full = env.LoadSamples(os.path.join("work","mycorpus","full.csv"), env["CORPORA_DIR"]+"/mycorpus/corpus_0_1.gz", CORPUS_NAME=dataset_name[0] )

    split_dataset_targets = [os.path.join("work", dataset_name[0], "split", "sample"+str(i)+".json") for i in range(0,dataset_name[1])]
    split_dataset_indices = [i for i in range(0,dataset_name[1])]
    targets_chunk_split = np.array_split(split_dataset_targets, env["NUM_CHUNKS"])
    indices_chunk_split = np.array_split(split_dataset_indices, env["NUM_CHUNKS"])

    sample_chunks = [env.SplitToChunks(list(cs), d_samples_full, CHUNK_INDICES = list(ci)) for cs, ci in zip(targets_chunk_split,indices_chunk_split)]
    

    chunk_embed_dict = defaultdict(list)
    for chunk in sample_chunks:
        for model_name in env["MODELS"]:
            model_results = []
            embed_names = [os.path.join("work",dataset_name[0],model_name,"embeddings",os.path.splitext(os.path.split(c.get_path())[-1])[0]+"_embed.pt") for c in chunk]
            if model_name in ["bert-large-uncased", "bert-base-uncased", "roberta-base", "roberta-large"]:
                chunk_embed_dict[model_name].append(env.EmbedBertlike(embed_names, chunk, MODEL_NAME=model_name, LAYERS=env["LAYERS"]))
            elif model_name in ["google/canine-c", "google/canine-s"]:
                chunk_embed_dict[model_name].append(env.EmbedCanine(embed_names,chunk,MODEL_NAME=model_name,LAYERS=["last"]))
            elif model_name == "general_character_bert":
                chunk_embed_dict[model_name].append(env.EmbedCBert(embed_names,chunk,MODEL_NAME=model_name))
    

    pred_results = defaultdict(list)
    for m_name, embeds in chunk_embed_dict.items():
        for chunk,embed in zip(sample_chunks,embeds):
            pred_names = [os.path.join("work",dataset_name[0],m_name,"preds",os.path.splitext(os.path.split(c.get_path())[-1])[0]+"_preds.json") for c in chunk]
            pred_results[m_name].append(env.Pred(pred_names, embed, MODEL_NAME=m_name, CHUNK=chunk, LAYERS=env["LAYERS"]))

  