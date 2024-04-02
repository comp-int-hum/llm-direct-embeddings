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
    ("MODELS","",["google/canine-c", "bert-large-uncased"]), #"bert-large-uncased", "roberta-large", "google/canine-s"]),
    ("LAYERS","",["last_four"]),#,"last, last_four", "first_three", "middle"]),
    ("DATA_PATH", "", "../corpora/orthocorpus.tar.gz"),
    ("DATASETS", "", ["HT_Prose_0_2", "GB_0_2"]),#["FCE-Subset", "GB_0_2", "HT_Prose_0_2"]),
    ("ALTERNATES_FILE","","data/britwords.csv"),
    ("CORPORA_DIR","","data"),
    ("RANDOM_STATE","", 10),
    ("NUM_CHUNKS","",25),
    ("MAX_LD","", 2),
    ("DEVICE", "", "cuda"), # cpu or cuda
    ("LDS_ANALYZE","",[2]),
    ("CUSTOM_LD","",0),
    ("USE_GRID","",False)
)


env = Environment(variables=vars, ENV=os.environ, #TARFLAGS="-c -z", TARSUFFIX=".tgz",
    #tools=["default", steamroller.generate]
    tools = [steamroller.generate],
    BUILDERS = {
	"LoadSamples" : Builder(action="python scripts/load_samples.py --input_file ${SOURCES[0]} --output_file ${TARGETS[0]} --max_ld ${MAX_LD} --custom_ld ${CUSTOM_LD} --dataset_name ${DATASET_NAME}"),
	"SplitToChunks" : Builder(action="python scripts/split_sample_chunks.py --input_file ${SOURCES[0]} --output_files ${TARGETS}"),
	"EmbedBertlike" : Builder(action="python scripts/get_tensors_bertlike.py ${SOURCES[0]} ${TARGETS[0]} --model ${MODEL_NAME} --layers ${LAYERS} --device ${DEVICE}"),
	"EmbedCanine" : Builder(action="python scripts/get_tensors_canine.py ${SOURCES[0]} ${TARGETS[0]} --model ${MODEL_NAME} --layers ${LAYERS} --device ${DEVICE}")
	}
)

#env.AddBuilder(
#    "LoadSamples",
#    "scripts/load_samples.py",
#    "--input_file ${SOURCES[0]} --output_file ${TARGETS[0]} --max_ld ${MAX_LD} --custom_ld ${CUSTOM_LD} --dataset_name ${DATASET_NAME}",
#)

#env.AddBuilder(
#    "SplitToChunks",
#    "scripts/split_sample_chunks.py",
#    "--input_file ${SOURCES[0]} --output_files ${TARGETS}"
#    )

#env.AddBuilder(
#    "EmbedBertlike",
#    "scripts/get_tensors_bertlike.py",
#    "${SOURCES[0]} ${TARGETS[0]} --model ${MODEL_NAME} --layers ${LAYERS} --device ${DEVICE}"
#)


#env.AddBuilder(
#    "EmbedCanine",
#    "scripts/get_tensors_canine.py",
#    "${SOURCES[0]} ${TARGETS[0]} --model ${MODEL_NAME} --layers ${LAYERS} --device ${DEVICE}"
#    )

#env.AddBuilder(
#    "EmbedCBert",
#    "scripts/get_tensors_cbert.py",
#    "${SOURCES} --model ${MODEL_NAME} --embeddings_out ${TARGETS}"
#    )

#env.AddBuilder(
#    "Pred",
#    "scripts/pred.py",
#    "${SOURCES[0]} ${TARGETS[0]} --model_name ${MODEL_NAME} --layers ${LAYERS} --alternates_file ${ALTERNATES_FILE} --max_ld ${MAX_LD}"
#    )

#env.AddBuilder(
#    "PredictionSummary",
#    "scripts/summarize_preds.py",
#    "${SOURCES} --accurate ${TARGETS[0]} --inaccurate ${TARGETS[1]} --summary ${TARGETS[2]} --layers ${LAYERS} --ld ${LD}"
#)

#env.AddBuilder(
#        "PredictionCSVSummary",
#        "scripts/csv_summary.py",
#        "${SOURCES} --summary_out ${TARGETS[0]}"
#)


# function for width-aware printing of commands
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print(s[:int(float(env["OUTPUT_WIDTH"]) / 2) - 2] + "..." + s[-int(float(env["OUTPUT_WIDTH"]) / 2) + 1:])
    else:
        print(s)


env['PRINT_CMD_LINE_FUNC'] = print_cmd_line


env.Decider("timestamp-newer")




#chunk into X pieces for use with -j --jobs (so can implicitly multicore over x processors)
#change to have a custom command line option that will force filebuilder instead of custom defined (to deal with graph )


for dataset_name in env["DATASETS"]:
    results_sums = []
    samples = env.LoadSamples(
        "work/${DATASET_NAME}_custom_${CUSTOM_LD}.json.gz",
        #"${DATA_PATH}/${DATASET_NAME}.tgz",
	"${DATA_PATH}",
        DATASET_NAME=dataset_name,
	CUSTOM_LD= env["CUSTOM_LD"]
    )
    chunks = env.SplitToChunks(
        ["work/${{DATASET_NAME}}_custom_${{CUSTOM_LD}}_chunk_{}.json.gz".format(i) for i in range(env["NUM_CHUNKS"])],
        samples,
        DATASET_NAME=dataset_name,
	CUSTOM_LD = env["CUSTOM_LD"]
    )




    chunk_embed_dict = defaultdict(list)
    for c_i,chunk in enumerate(chunks):
        for model_name in env["MODELS"]:
            if model_name in ["bert-large-uncased", "bert-base-uncased", "roberta-base", "roberta-large"]:
                chunk_embed_dict[model_name].append(env.EmbedBertlike("work/${DATASET_NAME}/${MODEL_NAME}/embeds/"+"chunk_embed_custom_${CUSTOM_LD}_"+str(c_i)+".json.gz", chunk, MODEL_NAME=model_name, LAYERS=env["LAYERS"], DATASET_NAME=dataset_name, CUSTOM_LD=env["CUSTOM_LD"]))

            elif model_name in ["google/canine-c", "google/canine-s"]:
                chunk_embed_dict[model_name].append(env.EmbedCanine("work/${DATASET_NAME}/${MODEL_NAME}/embeds/"+"chunk_embed_custom_${CUSTOM_LD}_"+str(c_i)+".json.gz",chunk,MODEL_NAME=model_name, LAYERS=["last"], DATASET_NAME=dataset_name, CUSTOM_LD=env["CUSTOM_LD"]))

"""
    pred_ld_results = defaultdict(lambda: defaultdict(list))
    for max_ld in env["LDS_ANALYZE"]:
        for m_name, embeds in chunk_embed_dict.items():
            for e_i,embed in enumerate(embeds):
                pred_ld_results[max_ld][m_name].append(env.Pred("work/${DATASET_NAME}/${MODEL_NAME}/preds/ld_${MAX_LD}_custom_${CUSTOM_LD}/chunk_pred"+str(e_i)+".json.gz", embed, MODEL_NAME=m_name, LAYERS=env["LAYERS"], DATASET_NAME=dataset_name, MAX_LD=max_ld, CUSTOM_LD=env["CUSTOM_LD"]))


    for ld, pred_results in pred_ld_results.items():
        for m_name, results in pred_results.items():
            if m_name in ["google/canine-c", "general_character_bert","google/canine-s"]:
	            results_sums.append(env.PredictionSummary([
                        "work/results/${DATASET_NAME}/${LD}/${MODEL_NAME}_accurate.csv",
                        "work/results/${DATASET_NAME}/${LD}/${MODEL_NAME}_inaccurate.csv",
                        "work/results/${DATASET_NAME}/${LD}_custom_${CUSTOM}/${MODEL_NAME}_summary.csv"],
                        results, MODEL_NAME=m_name, DATASET_NAME=dataset_name, LAYERS="last", LD=ld, CUSTOM=env["CUSTOM_LD"]))
            else:
                for layer in env["LAYERS"]:
                    results_sums.append(env.PredictionSummary([
                        "work/results/${DATASET_NAME}/${LD}/${MODEL_NAME}_${LAYERS}_accurate.csv",
                        "work/results/${DATASET_NAME}/${LD}/${MODEL_NAME}_${LAYERS}_inaccurate.csv",
                        "work/results/${DATASET_NAME}/${LD}_custom_${CUSTOM}/${MODEL_NAME}_${LAYERS}_summary.csv"],
                        results, MODEL_NAME=m_name, DATASET_NAME=dataset_name, LAYERS=layer, LD=ld, CUSTOM=env["CUSTOM_LD"]))

    env.PredictionCSVSummary(["work/results/${DATASET_NAME}/summary_custom_${CUSTOM}.csv"], [s[2] for s in results_sums], DATASET_NAME=dataset_name, CUSTOM=env["CUSTOM_LD"])
"""