import argparse
import sys
import torch
from utility.corpus_utils import maskSample, genOCRError
from utility.pred_utils import fetchMaxCSandMinEucinMaxLD, LAYER_LOOKUP

from transformers import AutoModel, AutoTokenizer
import os

import logging
import gzip
import json
import tarfile

import math

import nlpaug.augmenter.char as nac


log_format = "%(asctime)s::%(filename)s::%(message)s"

logging.basicConfig(level='INFO', format=log_format)


#combine embeddings? Train over known ground and created composites?

def get_hidden_states(encoded, token_ids_word, model, layers, layer_names, device):
    with torch.no_grad():
        logging.info("Processing sequence of shape %s", encoded["input_ids"].shape)
        output = model(**{k : v.to(device) for k, v in encoded.items()})
    states = output.hidden_states

    instance_layers = {}
    for instance_num in range(encoded["input_ids"].shape[0]):
        instance_layers[instance_num] = {}
        for layer_set, layer_name in zip(layers, layer_names):
            output = torch.stack([states[i][instance_num] for i in layer_set]).sum(0).squeeze()
            word_tokens_output = output[token_ids_word[instance_num]]
            instance_layers[instance_num][layer_name] = word_tokens_output.tolist()
    return instance_layers


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
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args, rest = parser.parse_known_args()

    device = torch.device(args.device)

#make sure to include N alternatives in prediction summaries

    logging.info("Loading model...")
    aug = nac.OcrAug()
    a_t = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True)
    model.to(device)
    layers = [LAYER_LOOKUP[l] for l in args.layers]

    logging.info("Model loaded")
    
    with gzip.open(args.chunk_json,"rt") as chunk_in, gzip.open(args.embeddings_out, "wt") as chunk_out:
        for line in chunk_in:
            json_sample = json.loads(line)
            logging.info("Processing sentence '%s'", json_sample["text"])
            
            for i, annotation in enumerate(json_sample["annotations"]):
                logging.info(
                    "\tProcessing observed token '%s' with standard form '%s'",
                    annotation["observed"],
                    annotation["standard"]
                )
                masked_sample = maskSample(
                    json_sample["text"],
                    annotation,
                    mask_token=a_t.mask_token
                ) #going to use mask as a place to insert candidates



                #initial encoding to locate masks after wordpiece tokenization
                encoded = a_t.encode(masked_sample, add_special_tokens=False)
                mask_index = encoded.index(a_t.mask_token_id)
                alts_encoded = [a_t.encode(alt, add_special_tokens=False) for alt in annotation["alts"]]

                orig_ns_encoded = a_t.encode(
                    annotation["observed"],
                    add_special_tokens=False
                )

                ground_encoded = a_t.encode(
                    annotation["standard"],
                    add_special_tokens=False
                )

                reverse_encoded = a_t.encode(
                    annotation["standard"][::-1],
                    add_special_tokens = False
                )

                error_encoded = a_t.encode(
                    aug.augment(annotation["standard"])[0],
                    #genOCRError(annotation["standard"]),
                    add_special_tokens = False)

                alt_lens = [len(a_e) for a_e in alts_encoded]
                alt_lens.append(len(orig_ns_encoded))
                alt_lens.append(len(ground_encoded))
                alt_lens.append(len(reverse_encoded))
                alt_lens.append(len(error_encoded))
                max_alt_len = max(alt_lens) if len(alt_lens) > 0 else 0


                #truncate around MASK
                if len(encoded) + max_alt_len >= args.maxlen-2:
                    print(len(encoded), max_alt_len)
                    budget_len = args.maxlen-2-max_alt_len
                    context_budget = math.floor(budget_len/2)
                    l_context_len  = len(encoded[0:mask_index])
                    r_context_len =  len(encoded[mask_index:])
                    print(budget_len)
                    print(context_budget)
                    print(l_context_len)
                    print(r_context_len)
                    if l_context_len > context_budget and r_context_len > context_budget:
                        l_start = l_context_len - context_budget
                        r_end = mask_index + context_budget

                    elif l_context_len < context_budget:
                        additional_budget = context_budget - l_context_len
                        l_start = 0
                        r_end = mask_index + context_budget + additional_budget
                        
                    elif r_context_len < context_budget:
                        additional_budget = context_budget - r_context_len
                        l_start = l_context_len - (context_budget+additional_budget)
                        r_end = mask_index + r_context_len
                    print(l_start, r_end)
                    encoded = encoded[l_start:r_end]
                    print(len(encoded), max_alt_len)
                    mask_index = encoded.index(a_t.mask_token_id)



                inputs = {
                    "input_ids" : [],
                    "attention_mask" : []
                }
                index_ranges = []


                orig_ns_inserted_ids, orig_index_range = insert_alt_id_at_mask(
                    orig_ns_encoded,
                    encoded,
                    mask_index
                )
                orig_ns_prepared = a_t.prepare_for_model(
                    ids=orig_ns_inserted_ids,
                    return_tensors="pt",
                    prepend_batch_axis=False
                )
                index_ranges.append(orig_index_range)
                inputs["input_ids"].append(orig_ns_prepared["input_ids"].tolist())
                inputs["attention_mask"].append(orig_ns_prepared["attention_mask"].tolist())
                
                ground_inserted_ids, ground_index_range = insert_alt_id_at_mask(
                    ground_encoded,
                    encoded,
                    mask_index
                )
                ground_prepared = a_t.prepare_for_model(
                    ids=ground_inserted_ids,
                    return_tensors="pt",
                    prepend_batch_axis=False
                )
                index_ranges.append(ground_index_range)
                inputs["input_ids"].append(ground_prepared["input_ids"].tolist())
                inputs["attention_mask"].append(ground_prepared["attention_mask"].tolist())

                reverse_inserted_ids, reverse_index_range = insert_alt_id_at_mask(
                    reverse_encoded,
                    encoded,
                    mask_index
                )
                reverse_prepared = a_t.prepare_for_model(
                    ids = reverse_inserted_ids,
                    return_tensors = "pt",
                    prepend_batch_axis = False
                )
                index_ranges.append(reverse_index_range)
                inputs["input_ids"].append(reverse_prepared["input_ids"].tolist())
                inputs["attention_mask"].append(reverse_prepared["attention_mask"].tolist())

                error_inserted_ids, error_index_range = insert_alt_id_at_mask(
                    error_encoded,
                    encoded,
                    mask_index
                )
                error_prepared = a_t.prepare_for_model(
                    ids = error_inserted_ids,
                    return_tensors = "pt",
                    prepend_batch_axis = False
                )
                index_ranges.append(error_index_range)
                inputs["input_ids"].append(error_prepared["input_ids"].tolist())
                inputs["attention_mask"].append(error_prepared["attention_mask"].tolist())


                lds = []
                for alt in annotation["alts"]:
                    lds.append(annotation["alts"][alt])
                    alt_encoded = a_t.encode(alt, add_special_tokens=False)
                    alt_inserted_ids, index_range = insert_alt_id_at_mask(alt_encoded, encoded, mask_index)
                    alt_prepared = a_t.prepare_for_model(ids=alt_inserted_ids, return_tensors="pt", prepend_batch_axis=False)
                    index_ranges.append(index_range)
                    inputs["input_ids"].append(alt_prepared["input_ids"].tolist())
                    inputs["attention_mask"].append(alt_prepared["attention_mask"].tolist())

                max_len = max([len(x) for x in inputs["input_ids"]])
                inputs["input_ids"] = torch.stack(
                    [torch.tensor(x + [0 for _ in range(max_len - len(x))]) for x in inputs["input_ids"]],
                    0
                )
                max_len = max([len(x) for x in inputs["attention_mask"]])
                inputs["attention_mask"] = torch.stack(
                    [torch.tensor(x + [0 for _ in range(max_len - len(x))]) for x in inputs["attention_mask"]],
                    0
                )
                outputs = get_hidden_states(
                    inputs,
                    index_ranges,
                    model,
                    layers,
                    args.layers,
                    device
                )
                json_sample["annotations"][i]["observed_embeddings"] = outputs[0]
                json_sample["annotations"][i]["standard_embeddings"] = outputs[1] #assume needed change
                json_sample["annotations"][i]["reverse_embeddings"] = outputs[2]
                json_sample["annotations"][i]["error_embeddings"] = outputs[3]

                
                json_sample["annotations"][i]["alts"] = {
                    alt : {"embed": emb, "LD": ld} for alt, emb, ld in zip(
                        json_sample["annotations"][i]["alts"],
                        #[outputs[j] for j in range(len(outputs) - 2)] #same
                        [outputs[j] for j in range(3, len(outputs))],
                        lds
                    )
                }                                 

            chunk_out.write(json.dumps(json_sample) + "\n")


