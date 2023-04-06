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
    ("MODELS","",["google/canine-c", "bert-large-uncased"]),
    #("MODELS","",["bert-large-uncased", "google/canine-c", "roberta-large", "general_character_bert"]),
    #("MODELS", "", ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large", "general_character_bert", "google/canine-c", "google/canine-s"]),
    #("LAYERS","", [[-1,-2,-3,-4], [-1], [1], [2], [3], [1,2,3], [6]]),
    ("LAYERS","",["last","last_four"]),#, "first_three", "middle"]),
    ("DATA_PATH", "", "corpora"),
    ("DATASETS", "", ["fce-released-dataset"]),#, "mycorpus"]),
    ("CORPORA_DIR","","corpora"),
    ("RANDOM_STATE","", 10),
    ("NUM_CHUNKS","",2000),
    ("MAX_LD","",3)
)


env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", steamroller.generate],
)

env.AddBuilder(
    "LoadSamples",
    "scripts/load_samples.py",
    "--input_file ${SOURCES[0]} --output_file ${TARGETS[0]} --max_ld ${MAX_LD}",
)

env.AddBuilder(
    "SplitToChunks",
    "scripts/split_sample_chunks.py",
    "--input_file ${SOURCES[0]} --output_files ${TARGETS}"
    )

env.AddBuilder(
    "EmbedBertlike",
    "scripts/get_tensors_bertlike.py",
    "${SOURCES[0]} ${TARGETS[0]} --model ${MODEL_NAME} --layers ${LAYERS}"
)


env.AddBuilder(
    "EmbedCanine",
    "scripts/get_tensors_canine.py",
    "${SOURCES[0]} ${TARGETS[0]} --model ${MODEL_NAME} --layers ${LAYERS}"
    )

env.AddBuilder(
    "EmbedCBert",
    "scripts/get_tensors_cbert.py",
    "${SOURCES} --model ${MODEL_NAME} --embeddings_out ${TARGETS}"
    )

env.AddBuilder(
    "Pred",
    "scripts/pred.py",
    "${SOURCES[0]} ${TARGETS[0]} --model_name ${MODEL_NAME} --layers ${LAYERS}"

    )

env.AddBuilder(
    "PredictionSummary",
    "scripts/summarize_preds.py",
    "${SOURCES} --outfile ${TARGETS[0]} --layers ${LAYERS} --model_name ${MODEL_NAME}"
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

    chunks = [chunks[0]]
    

    chunk_embed_dict = defaultdict(list)
    for c_i,chunk in enumerate(chunks):
        for model_name in env["MODELS"]:
            if model_name in ["bert-large-uncased", "bert-base-uncased", "roberta-base", "roberta-large"]:
                chunk_embed_dict[model_name].append(env.EmbedBertlike("work/${DATASET_NAME}/${MODEL_NAME}/embeds/"+"chunk_embed"+str(c_i)+".json.gz", chunk, MODEL_NAME=model_name, LAYERS=env["LAYERS"], DATASET_NAME=dataset_name))

            elif model_name in ["google/canine-c", "google/canine-s"]:
                chunk_embed_dict[model_name].append(env.EmbedCanine("work/${DATASET_NAME}/${MODEL_NAME}/embeds/"+"chunk_embed"+str(c_i)+".json.gz",chunk,MODEL_NAME=model_name,LAYERS=["last"], DATASET_NAME=dataset_name))
            #elif model_name == "general_character_bert":
                #chunk_embed_dict[model_name].append(env.EmbedCBert(embed_names,chunk,MODEL_NAME=model_name))
    

    pred_results = defaultdict(list)
    for m_name, embeds in chunk_embed_dict.items():
        for e_i,embed in enumerate(embeds):
            pred_results[m_name].append(env.Pred("work/${DATASET_NAME}/${MODEL_NAME}/preds/chunk_pred"+str(e_i)+".json.gz", embed, MODEL_NAME=m_name, LAYERS=env["LAYERS"], DATASET_NAME=dataset_name))

    results_sums = []
    for m_name, results in pred_results.items():
        results_sums.append(env.PredictionSummary("work/results/${DATASET_NAME}/${MODEL_NAME}_results.csv", results, MODEL_NAME=m_name, DATASET_NAME=dataset_name, LAYERS=env["LAYERS"]))
