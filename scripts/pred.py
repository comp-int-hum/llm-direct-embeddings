import torch
import logging
import json
import argparse
import gzip
import csv

from utility.pred_utils import LAYER_LOOKUP
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from dataclasses import dataclass

from bisect import insort, bisect

@dataclass
class Alt:
        LD: float
        CD: float
        alt: str
        def __lt__(self,other):
                return (self.CD > other.CD) and (self.LD <= other.LD)
@dataclass
class Alt_MO:
        LD: float
        CD: float
        alt: str
        def __lt__(self, other):
                return self.CD < other.CD

log_format = "%(asctime)s::%(filename)s::%(message)s"

logging.basicConfig(level='INFO', format=log_format)

def checkWordEquality(standard, alt, alt_dict):
        if alt == standard:
                return True
        if standard in alt_dict:
                if alt_dict[standard] == alt:
                        return True
        #if lemmatizer.lemmatize(alt) == lemmatizer.lemmatize(standard):
                #return True

        return False

def permissiveEquality(standard, alt, alt_dict):
        if checkWordEquality(standard, alt, alt_dict):
                return True
        if lemmatizer.lemmatize(alt) == lemmatizer.lemmatize(standard):
                return True
        if stemmer.stem(alt) == stemmer.stem(standard):
                return True
        return False

def checkStandardInAlts(standard, alts, alt_dict):
        if standard in alts:
                return True
        if standard in alt_dict:
                if alt_dict[standard] in alts:
                        return True
        return False


#combine embeddings? Train over known ground and created composites?

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("chunk_embed", help="Samples from chunk embedded")
        parser.add_argument("pred_out", help="Outfile of predictions")
        parser.add_argument("--model_name", dest="model_name", help="A model name of a bertlike model")
        parser.add_argument("--layers", nargs="+", dest="layers", help="Layers")
        parser.add_argument("--alternates_file", dest="alternates_file", help="A 2 column CSV file of alternate (say brit/us english) correct spellings")
        parser.add_argument("--max_ld", type=int, default=3, help="Max ld for prediction, up to 3")
        parser.add_argument("--ignore_an", type=bool, default=False)

        args, rest = parser.parse_known_args()

        conversion_dict = {}
        with open(args.alternates_file, "rt") as alts_in:
                alts_reader = csv.reader(alts_in)
                for line in alts_reader:
                        conversion_dict[line[0]] = line[1]
                if args.ignore_an:
                        conversion_dict["and"]="an'"
                        


        if args.model_name in ["google/canine-c", "general_character_bert","google/canine-s"]:
                args.layers = ["last"]


        with gzip.open(args.chunk_embed, "rt") as embed_in, gzip.open(args.pred_out, "wt") as json_out:
                for e_l in embed_in:
                        embeds = json.loads(e_l)
                        total_n_annotations = len(embeds["annotations"]) + len(embeds["other_ided_ns"])

                        out_l = []
                        for annotation in embeds["annotations"]:
                                final_pred = {l:{"alt":None, "CD": 0, "LD":1000, "acc":False, "acc_permissive":False, "std_cd": torch.cosine_similarity(torch.Tensor(annotation["standard_embeddings"][l]).mean(dim=0), torch.Tensor(annotation["observed_embeddings"][l]).mean(dim=0), dim=0).item()} for l in args.layers}
                                final_pred_only_model = {l:{"alt":None, "CD": 0, "LD":1000, "acc":False, "acc_permissive":False} for l in args.layers}
                                #alt_preds = {}
                                ap_dict = {}
                                alt_preds = {l:[] for l in args.layers}
                                standard_in_alts = checkStandardInAlts(annotation["standard"], annotation["alts"], conversion_dict)
                                for alt, embed in annotation["alts"].items():
                                        if embed["LD"] <= args.max_ld:
                                                ap_dict[alt] = {l:{} for l in args.layers}
                                                #alt_preds[alt] = {l:{} for l in args.layers}
                                                for layer in args.layers:
                                                        pred = Alt(embed["LD"], torch.cosine_similarity(torch.Tensor(annotation["observed_embeddings"][layer]).mean(dim=0), torch.Tensor(embed["embed"][layer]).mean(dim=0),dim=0).item(), alt)
                                                        insort(alt_preds[layer], pred)
                                                        
                                std_ranks = {l: 0 for l in args.layers}
                                for l in args.layers:
                                        if len(alt_preds[l]) > 0:
                                                std_pred = Alt(annotation["std_obs_ld"], final_pred[l]["std_cd"], annotation["standard"])
                                                final_pred[l] = {"alt":alt_preds[l][0].alt, "CD":alt_preds[l][0].CD, "LD":alt_preds[l][0].LD,
                                                  "acc":checkWordEquality(annotation["standard"], alt_preds[l][0].alt, conversion_dict),
                                                  "acc_permissive": permissiveEquality(annotation["standard"], alt_preds[l][0].alt,conversion_dict),
                                                  "std_cd":torch.cosine_similarity(torch.Tensor(annotation["standard_embeddings"][l]).mean(dim=0), torch.Tensor(annotation["observed_embeddings"][l]).mean(dim=0), dim=0).item()}
                                                std_ranks[l] = bisect(alt_preds[l], std_pred) + 1

                                mo_std_ranks = {l: 0 for l in args.layers}
                                for l in args.layers:
                                        alt_preds_mo = []
                                        if len(alt_preds[l]) > 0:
                                                for ap in alt_preds[l]:
                                                        insort(alt_preds_mo, Alt_MO(ap.LD, ap.CD, ap.alt))
                                                final_pred_only_model[l] = {"alt":alt_preds_mo[0].alt, "CD":alt_preds_mo[0].CD, "LD":alt_preds_mo[0].LD, "acc": checkWordEquality(annotation["standard"], alt_preds_mo[0].alt, conversion_dict), "acc_permissive": permissiveEquality(annotation["standard"], alt_preds_mo[0].alt, conversion_dict)}
                                                std_pred_mo = Alt_MO(annotation["std_obs_ld"],final_pred[l]["std_cd"], annotation["standard"])
                                                mo_std_ranks[l] = bisect(alt_preds_mo, std_pred_mo) + 1                   
                                                
                                                
                                        #cs_sorted = sorted(alt_preds[l], key=lambda x: x.CD)
                                        #if len(cs_sorted) > 0:
                                        #        final_pred_only_model[l] = {"alt": cs_sorted[0].alt, "CD":cs_sorted[0].CD, "LD":cs_sorted[0].LD,
                                        #        "acc": checkWordEquality(annotation["standard"], cs_sorted[0].alt, conversion_dict),
                                        #        "acc_permissive": permissiveEquality(annotation["standard"], cs_sorted[0].alt, conversion_dict)}
                                                                    
                                
 
                                out_l.append({"preds": ap_dict, "final_pred":final_pred, "final_pred_model_only":final_pred_only_model, "observed":annotation["observed"], "standard": annotation["standard"], "alt_present":int(standard_in_alts), "num_alts":len(annotation["alts"]), "total_ns_in_sample":total_n_annotations, "std_ranks":std_ranks, "mo_std_ranks": mo_std_ranks})


                        json_out.write(json.dumps(out_l) + "\n")


