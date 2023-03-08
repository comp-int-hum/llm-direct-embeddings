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

# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle


vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 5000),
    ("MODELS","",["google/canine-c"]),
    #("MODELS", "", ["bert-base-uncased", "bert-large-uncased", "robert-base", "roberta-large", "general_character_bert", "google/canine-c", "google/canine-s"]),
    ("LAYERS","", [[-1,-2,-3,-4], [-1], [1], [2], [3], [1,2,3], [6]]),
    ("DATASETS", "", ["fce-released-dataset","mycorpus"]),
    ("CORPORA_DIR","","corpora"),
    ("RANDOM_STATE","", 10),
    ("NUM_CHUNKS","",50),
    ("MAX_LD","",3)
)


env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", steamroller.generate],
)

env.AddBuilder(
    "LoadSamples",
    "scripts/load_samples.py",
    "${DATASET_NAME} --corpus_dir ${CORPUS_DIR} --chunks ${NUM_CHUNKS} --outfiles ${TARGETS}",
)

env.AddBuilder(
    "RunBertlikePred",
    "scripts/pred_bertlike.py",
    "${SOURCES[0]} --model ${MODEL_NAME} --pred_out ${TARGETS[0]} --layers ${LAYERS} --max_ld ${MAX_LD}"
)


env.AddBuilder(
    "RunCBERTPred",
    "scripts/pred_cbert.py",
    "${SOURCES[0]} --pred_out ${TARGETS[0]} --max_ld ${MAX_LD}"
)

env.AddBuilder(
    "RunCaninePred",
    "scripts/pred_canine.py",
    "${SOURCES[0]} --model ${MODEL_NAME} --pred_out ${TARGETS[0]}, --layers ${LAYERS} --max_ld ${MAX_LD}"

    )

"""
env.AddBuilder(
    "Tokenize",
    "scripts/tokenizer.py",
    "${SOURCES[0]} --outfile ${TARGETS[0]} --model ${MODEL_NAME}"
)

env.AddBuilder(
    "GetWordEmbeddings",
    "scripts/get_embeddings.py",
    "${SOURCES[0]} --embeddings_out ${TARGETS[0]} --classifications_out ${TARGETS[1]}  --model ${MODEL_NAME} --layers ${LAYERS}"

    )

env.AddBuilder(
    "GetCharBertEmbeddings",
    "scripts/get_embeddings_cbert.py",
    "${SOURCES[0]} --embeddings_out ${TARGETS[0]} --classifications_out ${TARGETS[1]} --model ${MODEL_NAME} --layers ${LAYERS}"

    )

env.AddBuilder(
    "TrainTestSplit",
    "scripts/train_split.py",
    "${SOURCES[0]} ${SOURCES[1]} --outputs ${TARGETS} --random_state ${RANDOM_STATE}"

    )

env.AddBuilder(
    "TrainModel",
    "scripts/train_model.py",
    "${SOURCES} --outputs ${TARGETS[0]} --random_state ${RANDOM_STATE}"
)

env.AddBuilder(
    "TestModel",
    "scripts/test_model.py",
    "${SOURCES} --outputs ${TARGETS[0]}"
)
"""

# function for width-aware printing of commands
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print(s[:int(float(env["OUTPUT_WIDTH"]) / 2) - 2] + "..." + s[-int(float(env["OUTPUT_WIDTH"]) / 2) + 1:])
    else:
        print(s)


env['PRINT_CMD_LINE_FUNC'] = print_cmd_line


env.Decider("timestamp-newer")


#chunk into X pieces for use with -j --jobs (so can implicitly multicore over x processors)
res = []
for dataset_name in env["DATASETS"]:
    s_chunks = ["work/"+dataset_name+"/samplec"+str(x)+".csv" for x in range(0,env["NUM_CHUNKS"])]
    samples = env.LoadSamples(s_chunks, [], [], DATASET_NAME = dataset_name, CORPUS_DIR = env["CORPORA_DIR"], NUM_CHUNKS = env["NUM_CHUNKS"])

    for model_name in env["MODELS"]:
        if model_name in ["bert-large-uncased", "bert-base-uncased", "roberta-base", "roberta-large"]:
            for layer in env["LAYERS"]:
                for i,schunk in enumerate(samples):
                    res.append(env.RunBertlikePred("work/${DATASET_NAME}/${MODEL_NAME}/${LAYERS}/pred"+str(i)+".csv", schunk, [], DATASET_NAME=dataset_name, MODEL_NAME=model_name, LAYERS=layer))
        elif model_name == "general_character_bert":
            for i, schunk in enumerate(samples):
                res.append(env.RunCBERTPred("work/${DATASET_NAME}/${MODEL_NAME}/pred"+str(i)+".csv", schunk, [], DATASET_NAME=dataset_name))
        elif model_name in ["google/canine-c", "google/canine-s"]:
            for layer in env["LAYERS"]:
                for i, schunk in enumerate(samples):
                    res.append(env.RunCaninePred("work/${DATASET_NAME}/${MODEL_NAME}/${LAYERS}/pred"+str(i)+".csv", schunk, [], DATASET_NAME=dataset_name, MODEL_NAME=model_name, LAYERS=layer))

