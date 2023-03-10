import argparse
import sys
import pandas as pd
import torch
from utility.corpus_utils import maskSample, loadBrownCorpusTree
from utility.pred_utils import fetchMaxCSandMinEucinMaxLD

from transformers import AutoModel, AutoTokenizer
import jellyfish
import os

import logging



log_format = "%(asctime)s::%(filename)s::%(message)s"

logging.basicConfig(level='INFO', format=log_format)
#scons -N 10 (or whatever) jobs
#then chunk data into 10 chunks, and iterate over it applying all variants eventually to each
#should auto parallelize 


#combine embeddings? Train over known ground and created composites?

def get_hidden_states(encoded, token_ids_word, model, layers):
    with torch.no_grad():
        output = model(**encoded)

    states = output.hidden_states
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    word_tokens_output = output[token_ids_word]
    return word_tokens_output.mean(dim=0)

def insert_alt_id_at_mask(alt_id, encoded, mask_index):
    len_alt_encoded = len(alt_id)
    orig_inserted_ids = encoded[:mask_index] + alt_id + encoded[mask_index:][1:]
    inserted_index_range_in_sent = [i for i in range(mask_index+1, mask_index+1+len_alt_encoded)]
    return orig_inserted_ids, inserted_index_range_in_sent


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("seglocs", help="CSV of segments")
    parser.add_argument("--maxlen", default=512,  help="Max unit (token) len for models")
    parser.add_argument("--pred_out", dest="outfile",  help="Outfile of predictions")
    parser.add_argument("--model", dest="model_name", help="A model name of a bertlike model")
    parser.add_argument("--max_ld", type=int, default=3,  help="Max LD")
    parser.add_argument("--layers", nargs="+", default=["-1"], dest="layers")
    parser.add_argument("--dictionary", default="brown")

    args, rest = parser.parse_known_args()

    s_df = pd.read_csv(args.seglocs)
    a_t = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True)
    layers = [int(l) for l in args.layers]
    bktree = loadBrownCorpusTree()


    s_df["MaskedSample"] = s_df.apply(maskSample, mask_token=a_t.mask_token, axis=1) #going to use mask as a place to insert candidates

    def predBertlike(row):

        encoded = a_t.encode(row["MaskedSample"], add_special_tokens=False)
        mask_index = encoded.index(a_t.mask_token_id)

        orig_ns_encoded = a_t.encode(row["NS"],add_special_tokens=False)
        orig_ns_inserted_ids, index_range = insert_alt_id_at_mask(orig_ns_encoded, encoded, mask_index)
        orig_ns_prepared = a_t.prepare_for_model(ids=orig_ns_inserted_ids, return_tensors="pt", prepend_batch_axis=True)
        orig_ns_vec = get_hidden_states(orig_ns_prepared, index_range, model, layers)

        ground_encoded = a_t.encode(row["Ground"],add_special_tokens=False)
        ground_inserted_ids, index_range = insert_alt_id_at_mask(ground_encoded, encoded, mask_index)
        ground_prepared = a_t.prepare_for_model(ids=ground_inserted_ids, return_tensors="pt", prepend_batch_axis=True)
        ground_vec = get_hidden_states(ground_prepared, index_range, model, layers)

        ns_ground_cs = torch.cosine_similarity(orig_ns_vec.reshape(1,-1), ground_vec.reshape(1,-1)).numpy().tolist()[0]
        ns_ground_ld = jellyfish.levenshtein_distance(row["Ground"], row["NS"])

        logging.info(row["Ground"])
        logging.info(row["NS"])
        

        alt_ids = [w for w in bktree.find(row["NS"], args.max_ld)] #maxld
        full_res_df = pd.DataFrame.from_records({"Alt":[a[1] for a in alt_ids], "LD":[a[0] for a in alt_ids]})

        def fullPreds(row):
            alt_inserted_ids, index_range = insert_alt_id_at_mask(a_t.encode(row["Alt"], add_special_tokens=False), encoded, mask_index)
            alt_inserted_prepared = a_t.prepare_for_model(ids=alt_inserted_ids, return_tensors="pt", prepend_batch_axis=True)
            alt_vec = get_hidden_states(alt_inserted_prepared, index_range, model, layers)
            
            return torch.cosine_similarity(orig_ns_vec.reshape(1,-1), alt_vec.reshape(1,-1)).numpy().tolist()[0], torch.cdist(orig_ns_vec.reshape(1,-1), alt_vec.reshape(1,-1)).numpy().tolist()[0][0]

        full_res_df[["CSSim","EucDist"]] = full_res_df.apply(fullPreds, axis=1, result_type="expand")
        full_res_df.to_csv(os.path.join(os.path.dirname(args.outfile),str(row.name)+".csv"))

        max_cs_in_min_ld, min_euc_in_min_ld = fetchMaxCSandMinEucinMaxLD(full_res_df)


        logging.info(max_cs_in_min_ld.Alt.to_numpy()[0])
        logging.info(max_cs_in_min_ld.CSSim.to_numpy()[0])

        max_cs = full_res_df[full_res_df.CSSim == full_res_df.CSSim.max()]
        logging.info(max_cs.Alt.to_numpy()[0])


        return ns_ground_cs, ns_ground_ld, max_cs_in_min_ld.Alt.to_numpy()[0], max_cs_in_min_ld.LD.to_numpy()[0], max_cs_in_min_ld.CSSim.to_numpy()[0], min_euc_in_min_ld.Alt.to_numpy()[0], min_euc_in_min_ld.LD.to_numpy()[0], min_euc_in_min_ld.EucDist.to_numpy()[0], max_cs.Alt.to_numpy()[0]

    s_df[["NS_GroundCS", "NS_GroundLD", "Pred", "PredLD","PredCS", "PredEuc", "PredEucLD", "PredEucEuc", "Max_CSAlt"]] = s_df.apply(predBertlike, axis=1, result_type="expand")
    s_df.to_csv(args.outfile)

    #get original located embedding
    #tokenize over masked with inserted candidates
    #prune to length if nesc (should happen after inserting candidate, as may be X tokens)








