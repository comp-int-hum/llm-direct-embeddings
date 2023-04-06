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
    ("DATA_PATH", "", "data"),
    ("DATASETS", "", ["fce-released-dataset", "mycorpus"]),
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
    "--input_file ${SOURCES[0]} --output_file ${TARGETS[0]}",
)

env.AddBuilder(
    "SplitToChunks",
    "scripts/split_sample_chunks.py",
    "--input_file ${SOURCES[0]} --output_files ${TARGETS}"
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



#num samples (max)
#handle null cases in actual scripts

#num_chunks
#so that would be num files
#but possibly a file would have one entry, or more




#chunk into X pieces for use with -j --jobs (so can implicitly multicore over x processors)

for dataset_name in env["DATASETS"]:
    samples = env.LoadSamples(
        "work/${DATASET_NAME}.json.gz",
        "${DATA_PATH}/${DATASET_NAME}.tgz",
        DATASET_NAME=dataset_name
    )
    chunks = env.SplitToChunks(
        ["work/${{DATASET_NAME}}_chunk_{}.json.gz".format(i) for i in range(env["NUM_CHUNKS"])],
        samples,
        DATASET_NAME=dataset_name
    )
    continue
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

            



"""


    s_chunks = ["work/"+dataset_name+"/samplec"+str(x)+".csv" for x in range(0,env["NUM_CHUNKS"])]
    samples = env.LoadSamples(s_chunks, [], [], DATASET_NAME = dataset_name, CORPUS_DIR = env["CORPORA_DIR"], NUM_CHUNKS = env["NUM_CHUNKS"])

    for model_name in env["MODELS"]:
        if model_name in ["bert-large-uncased", "bert-base-uncased", "roberta-base", "roberta-large"]:
            embeddings = 


            for layer in env["LAYERS"]:
                r = []
                for i,schunk in enumerate(samples):

                    r.append(env.RunBertlikePred("work/${DATASET_NAME}/${MODEL_NAME}/${LAYERS}/pred"+str(i)+".gz", schunk, DATASET_NAME=dataset_name, MODEL_NAME=model_name, LAYERS=layer))
                    r2.append(r)
                res.append(env.EvalResults("work/${DATASET_NAME}/${MODEL_NAME}/${LAYERS}/results.csv", r, DATASET_NAME=dataset_name, MODEL_NAME=model_name, LAYERS=layer))
        elif model_name == "general_character_bert":
            r = []
            for i, schunk in enumerate(samples):
                r.append(env.RunCBERTPred("work/${DATASET_NAME}/${MODEL_NAME}/pred"+str(i)+".gz", schunk, DATASET_NAME=dataset_name, MODEL_NAME=model_name))
                r2.append(r)
            res.append(env.EvalResultsChar("work/${DATASET_NAME}/${MODEL_NAME}/results.csv", r, DATASET_NAME=dataset_name, MODEL_NAME=model_name))
        elif model_name in ["google/canine-c", "google/canine-s"]:
            r = []
            for layer in env["LAYERS"]:
                for i, schunk in enumerate(samples):
                    r.append(env.RunCaninePred("work/${DATASET_NAME}/${MODEL_NAME}/${LAYERS}/pred"+str(i)+".gz", schunk, DATASET_NAME=dataset_name, MODEL_NAME=model_name, LAYERS=layer))
                    r2.append(r)
                res.append(env.EvalResultsChar("work/${DATASET_NAME}/${MODEL_NAME}/${LAYERS}/results.csv", r, DATASET_NAME=dataset_name, MODEL_NAME=model_name, LAYERS=layer))


"""
