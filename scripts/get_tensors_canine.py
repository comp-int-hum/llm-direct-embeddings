import argparse
import sys
import torch
from utility.pred_utils import fetchMaxCSandMinEucinMaxLD, LAYER_LOOKUP

from transformers import CanineModel

import os

import logging
import json



log_format = "%(asctime)s::%(filename)s::%(message)s"

logging.basicConfig(level='INFO', format=log_format)


#combine embeddings? Train over known ground and created composites?

def get_hidden_states(encoded, char_span, model, layers, layer_names):
    with torch.no_grad():
        output = model(encoded)

    states = output.hidden_states
    layer_d = {}
    for layer_set, layer_name in zip(layers,layer_names):
        output = torch.stack([states[i] for i in layer_set]).sum(0).squeeze()
        char_encoded_output = output[char_span[0]:char_span[1]]
        layer_d[layer_name] = char_encoded_output
    return layer_d

def insert_alt_seq_at_index(orig, alt_seq, s_i, e_i):
    ins_str = orig[0:s_i] + alt_seq + orig[e_i:]
    new_final_index = s_i + len(alt_seq)
    encoded = torch.tensor([[ord(c) for c in ins_str]])
    return encoded, new_final_index


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_json", nargs="+", help="List of CSV sample chunks")
    parser.add_argument("--maxlen", default=512,  help="Max unit (token) len for models")
    parser.add_argument("--embeddings_out", nargs="+", dest="outfiles",  help="Embedding names")
    parser.add_argument("--model", dest="model_name", help="A model name of a bertlike model")
    parser.add_argument("--layers", nargs="+", default=["last"], dest="layers")

    args, rest = parser.parse_known_args()


    
    model = CanineModel.from_pretrained(args.model_name,output_hidden_states=True)
    layers = [LAYER_LOOKUP[l] for l in args.layers]

    for in_samp,out_embed in zip(args.chunk_json, args.outfiles):
        with open(in_samp, "r") as j_in:
            json_sample = json.load(j_in)


        s_i = json_sample["i"]
        e_i = json_sample["i"] + len(json_sample["NS"])

        orig_str = json_sample["sample"]
        print(len(orig_str))
        orig_encoded = torch.tensor([[ord(c) for c in orig_str]])
        ground_inserted_encoded, g_i_e_i = insert_alt_seq_at_index(orig_str, json_sample["Ground"], s_i, e_i)

        orig_embed = get_hidden_states(orig_encoded, [s_i,e_i], model, layers, args.layers)
        ground_embed = get_hidden_states(ground_inserted_encoded, [s_i,g_i_e_i], model, layers, args.layers)


        logging.info("Sample: "+json_sample["sample"])
        logging.info("Ground: "+json_sample["Ground"])
        logging.info("NS: "+json_sample["NS"])
        logging.info("Num Alts: " + str(len(json_sample["Alts"])))

        out_d = {json_sample["NS"]: orig_embed, json_sample["Ground"]: ground_embed}
        print(out_d)
        for alt in json_sample["Alts"]:
            alt_inserted_encoded, new_ei = insert_alt_seq_at_index(orig_str, alt, s_i, e_i)
            alt_embed = get_hidden_states(alt_inserted_encoded, [s_i,new_ei], model, layers, args.layers)
            out_d[alt] = alt_embed


        torch.save(out_d,out_embed)
