import argparse
import re
import gzip
import json
import tarfile
from bs4 import BeautifulSoup
from utility.corpus_utils import loadBrownCorpusTree, loadBrownWNCorpusTree, loadSWTree, loadBrownCustomTree
from nltk.corpus import stopwords

from nltk.corpus import wordnet as wn
import jellyfish
from utility.custom_ld import levenshtein

def loadCorpus(fname, ifname, sent_sep=True):
    with tarfile.open(fname, "r") as tifd:
        print([t for t in tifd.getnames()])
        name = [t for t in tifd.getnames() if ifname in t][0]
        member = tifd.getmember(name)
        ifd = tifd.extractfile(member)
        print(member.name)
        print(ifname+".jsonl")

        for line in ifd:
            j = json.loads(line)
            print(j)
            item = {
                "text" : j["text"],
                "annotations" : j["annotations"],
                "other_ided_ns": []
            }
            yield item
            #for obs, props in j["annotations"].items():
            #    for start in props["i"]:
            #        if props["Std"] != "{}":
            #            end = start + len(obs)
            #            item["annotations"].append(
            #                {
            #                    "start" : start,
            #                    "end" : end,
            #                    "observed" : item["text"][start:end],
            #                    "standard" : props["Std"],
            #                    "ocr" : props.get("OCR", False)
            #                }
            #            )
            #    yield item


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", dest="input_file",  help="Input file")
    parser.add_argument("--output_file", dest="output_file",  help="Output file")
    parser.add_argument("--max_ld", dest="max_ld", type=int, default=3,  help="Max LD")
    parser.add_argument("--custom_ld", type=int, default=0, help="use weighted LD")
    parser.add_argument("--dataset_name", help="subset to load")
    args, rest = parser.parse_known_args()

    #brown_wn_tree = loadBrownWNCorpusTree()
    brown_tree = loadBrownCorpusTree() if not args.custom_ld else loadBrownCustomTree()
    #sw_tree = loadSWTree()

    with gzip.open(args.output_file, "wt") as ofd:

        for item in loadCorpus(args.input_file, args.dataset_name):
            print(item["text"])
            offset = 0
            for sentence_match in re.finditer(r"(.+?(?:(?:(?:\.|\?|\!)\s+)|$))", item["text"], re.S):
                sentence = sentence_match.group(1)
                sentence_item = {
                    "text" : sentence.lower(),
                    "annotations" : [],
                    "other_ided_ns": []                }
                new_offset = offset + len(sentence)
                for other_ns in item["other_ided_ns"]:
                    if all([
                            other_ns["start"] < new_offset,
                            other_ns["start"] < offset,
                            other_ns["end"] < new_offset
                        ]):
                            sentence_item["other_ided_ns"].append({
                                "start":other_ns["start"],
                                "end":other_ns["end"]
                                })


                for annotation in item["annotations"]:
                    if all(
                            [
                                annotation["start"] < new_offset,
                                annotation["start"] > offset,
                                annotation["end"] < new_offset,
                                annotation["observed"] != annotation["standard"]
                            ]
                    ):
                        sentence_item["annotations"].append(
                            {
                                "start" : annotation["start"] - offset,
                                "end" : annotation["end"] - offset,
                                "observed" : annotation["observed"].lower(),
                                "standard" : annotation["standard"].lower(),
                                "ocr" : annotation["ocr"],
                                "std_obs_ld": jellyfish.levenshtein_distance(annotation["observed"].lower(), annotation["standard"].lower()) if not args.custom_ld else levenshtein(annotation["observed"].lower(), annotation["standard"].lower()),
                                #"alts": {a[1]: a[0] for a in brown_wn_tree.find(annotation["observed"].lower(), args.max_ld)}
                                "alts": {a[1]: a[0] for a in brown_tree.find(annotation["observed"].lower(), args.max_ld) if len(wn.synsets(a[1])) > 0 or a[1] in stopwords.words("english")}                     

                            }
                        )
                offset = new_offset
                if len(sentence_item["annotations"]) > 0:
                    ofd.write(json.dumps(sentence_item) + "\n")
