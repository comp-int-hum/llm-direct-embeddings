import argparse
import sys
import torch
from utility.corpus_utils import maskSample
from utility.pred_utils import fetchMaxCSandMinEucinMaxLD, LAYER_LOOKUP

from transformers import AutoModel, AutoTokenizer
import os

import logging
import gzip
import json
import tarfile



log_format = "%(asctime)s::%(filename)s::%(message)s"

logging.basicConfig(level='INFO', format=log_format)


#combine embeddings? Train over known ground and created composites?

def get_hidden_states(encoded, token_ids_word, model, layers, layer_names):
    with torch.no_grad():
        output = model(**encoded)
    states = output.hidden_states
    layer_d = {}
    for layer_set, layer_name in zip(layers,layer_names):
        output = torch.stack([states[i] for i in layer_set]).sum(0).squeeze()
        word_tokens_output = output[token_ids_word]
        layer_d[layer_name] = word_tokens_output.tolist()
    return layer_d

def insert_alt_id_at_mask(alt_id, encoded, mask_index):
    len_alt_encoded = len(alt_id)
    orig_inserted_ids = encoded[:mask_index] + alt_id + encoded[mask_index:][1:]
    inserted_index_range_in_sent = [i for i in range(mask_index+1, mask_index+1+len_alt_encoded)]
    return orig_inserted_ids, inserted_index_range_in_sent


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_json", help="Gzed chunk")
    parser.add_argument("embeddings_out", help="Chunk embedding name")
    parser.add_argument("--maxlen", default=512,  help="Max unit (token) len for models")
    parser.add_argument("--model", dest="model_name", help="A model name of a bertlike model")
    parser.add_argument("--layers", nargs="+", default=["last"], dest="layers")

    args, rest = parser.parse_known_args()


    
    a_t = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True)
    layers = [LAYER_LOOKUP[l] for l in args.layers]

    with gzip.open(args.chunk_json,"rt") as chunk_in, gzip.open(args.embeddings_out, "wt") as chunk_out:
        for line in chunk_in:
            json_sample = json.loads(line)
            print(json_sample)


            for annotation in json_sample["annotations"]:

                masked_sample = maskSample(json_sample["text"], annotation, mask_token = a_t.mask_token) #going to use mask as a place to insert candidates
                #initial encoding to locate masks after wordpiece tokenization
                encoded = a_t.encode(masked_sample, add_special_tokens=False)
                mask_index = encoded.index(a_t.mask_token_id)
                sample_embeds = {"embeds":[]}


                orig_ns_encoded = a_t.encode(annotation["observed"],add_special_tokens=False)
                orig_ns_inserted_ids, index_range = insert_alt_id_at_mask(orig_ns_encoded, encoded, mask_index)
                orig_ns_prepared = a_t.prepare_for_model(ids=orig_ns_inserted_ids, return_tensors="pt", prepend_batch_axis=True)

                ground_encoded = a_t.encode(annotation["standard"],add_special_tokens=False)
                ground_inserted_ids, index_range = insert_alt_id_at_mask(ground_encoded, encoded, mask_index)
                ground_prepared = a_t.prepare_for_model(ids=ground_inserted_ids, return_tensors="pt", prepend_batch_axis=True)

                logging.info("Sample: "+json_sample["text"])
                logging.info("Ground: "+annotation["standard"])
                logging.info("NS: "+annotation["observed"])
                logging.info("Num Alts: " + str(len(annotation["alts"])))

                out_d = {"observed":{"token":annotation["observed"], "embed": get_hidden_states(orig_ns_prepared, index_range, model, layers, args.layers)}, "standard":{"token": annotation["standard"], "embed": get_hidden_states(ground_prepared, index_range, model, layers, args.layers)}, "alts":[]}
                for alt in annotation["alts"]:
                    alt_encoded = a_t.encode(alt, add_special_tokens=False)
                    alt_inserted_ids, index_range = insert_alt_id_at_mask(alt_encoded, encoded, mask_index)
                    alt_prepared = a_t.prepare_for_model(ids=alt_inserted_ids, return_tensors="pt", prepend_batch_axis=True)
                    out_d["alts"].append( {"token": alt, "embed": get_hidden_states(alt_prepared, index_range, model, layers, args.layers), "LD": annotation["alts"][alt]})

                sample_embeds["embeds"].append(out_d)


            chunk_out.write(json.dumps(sample_embeds) + "\n")

