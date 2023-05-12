import argparse
import sys
import torch
from utility.pred_utils import LAYER_LOOKUP

from transformers import CanineModel, CanineTokenizer

import os

import logging
import json
import gzip



log_format = "%(asctime)s::%(filename)s::%(message)s"

logging.basicConfig(level='INFO', format=log_format)


#combine embeddings? Train over known ground and created composites?

def get_hidden_states(encoded, start_index, char_end_indices, model, layers, layer_names, device):
    with torch.no_grad():
        logging.info("Processing sequence of shape %s", encoded["input_ids"].shape)
        output = model(**{k : v.to(device) for k, v in encoded.items()})

    states = output.hidden_states

    instance_layers = {}
    for instance_num in range(encoded["input_ids"].shape[0]):
        instance_layers[instance_num] = {}
        for layer_set, layer_name in zip(layers, layer_names):
            output = torch.stack([states[i][instance_num] for i in layer_set]).sum(0).squeeze()
            char_encoded_output = output[start_index:char_end_indices[instance_num]]
            instance_layers[instance_num][layer_name] = char_encoded_output.tolist()
    return instance_layers
    

def insert_alt_seq_at_index(orig, alt_seq, s_i, e_i):
    ins_str = orig[0:s_i] + alt_seq + orig[e_i:]
    new_final_index = s_i + len(alt_seq)
    #encoded = torch.tensor([[ord(c) for c in ins_str]])
    #return encoded, new_final_index
    return ins_str, new_final_index


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

    logging.info("Loading model...")
    model = CanineModel.from_pretrained(args.model_name,output_hidden_states=True)
    t_c = CanineTokenizer.from_pretrained(args.model_name)
    layers = [LAYER_LOOKUP[l] for l in args.layers]
    model.to(device)
    logging.info("Model loaded")

    #RETURN SIZE HANDLING

    with gzip.open(args.chunk_json,"rt") as chunk_in, gzip.open(args.embeddings_out, "wt") as chunk_out:
        for line in chunk_in:
            json_sample = json.loads(line)
            logging.info("Processing sentence '%s'", json_sample["text"]) 

            try:
                for i, annotation in enumerate(json_sample["annotations"]):


                    logging.info(
                        "\tProcessing observed token '%s' with standard form '%s'",
                        annotation["observed"],
                        annotation["standard"]
                    )

                    

                    ground_inserted, g_i_e_i = insert_alt_seq_at_index(json_sample["text"], annotation["standard"], annotation["start"], annotation["end"])
                    inputs = [json_sample["text"], ground_inserted]
                    end_indices = [annotation["end"]+1, g_i_e_i+1] #CLS token

                    lds = []
                    for alt in annotation["alts"]:
                        lds.append(annotation["alts"][alt])
                        alt_inserted, new_ei = insert_alt_seq_at_index(json_sample["text"], alt, annotation["start"], annotation["end"])
                        inputs.append(alt_inserted)
                        end_indices.append(new_ei+1)

                    encoded_inputs = t_c(inputs, padding=True, truncation=True, return_tensors="pt")

                    outputs = get_hidden_states(encoded_inputs,
                        annotation["start"] + 1,
                        end_indices,
                        model,
                        layers,
                        args.layers,
                        device,
                    )

                    json_sample["annotations"][i]["observed_embeddings"] = outputs[0]
                    json_sample["annotations"][i]["standard_embeddings"] = outputs[1]

                    json_sample["annotations"][i]["alts"] = {
                        alt : {"embed": emb, "LD": ld} for alt, emb, ld in zip(
                            json_sample["annotations"][i]["alts"],
                            [outputs[j] for j in range(2,len(outputs))],
                            lds
                        )
                    }


                chunk_out.write(json.dumps(json_sample) + "\n")
            except RuntimeError:
                print(json_sample["text"])
                print(len(annotation["alts"]))
                print(encoded_inputs.shape)
                input()