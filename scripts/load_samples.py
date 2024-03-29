import argparse
import re
import gzip
import json
import tarfile
from bs4 import BeautifulSoup
from utility.corpus_utils import loadBrownCorpusTree, loadBrownWNCorpusTree, loadSWTree, loadBrownCustomTree
from nltk.corpus import stopwords

from nltk.corpus import wordnet as wn


def loadCorpus(fname, sent_sep=True):
    with tarfile.open(fname, "r") as tifd:
        for member in [m for m in tifd.getmembers() if m.isfile()]:
            ifd = tifd.extractfile(member)
            if member.name.endswith("xml"):
                print(member.name)
                soup = BeautifulSoup(ifd, "xml")
                for ca in soup.find_all("coded_answer"):
                    for p in ca.find_all("p"):
                        item = {
                            "text" : "",
                            "annotations" : [],
                            "other_ided_ns": []
                        }
                        for elem in p:
                            if elem.name == "NS" and elem.i and elem.c and elem["type"] in ["S", "SA", "SX"]:
                                all_i = elem.find_all("i")
                                all_c = elem.find_all("c")
                                #if len(all_i) > 0 and len(all_c) > 1:
                                #    print(all_i[-1].text)
                                #    print(all_c[-1].text)

                                    #input()
                                start = len(item["text"])
                                end = start + len(all_i[-1].text)
                                item["text"] += all_i[-1].text
                                item["annotations"].append(
                                    {
                                        "start" : start,
                                        "end" : end,
                                        "standard" : all_c[-1].text,
                                        "observed" : all_i[-1].text,
                                        "ocr" : False
                                    }
                                )
                            elif elem.name == "NS":
                                start = len(item["text"])
                                if elem.i:
                                    all_i = elem.find_all("i")
                                    item["text"] += all_i[-1].text
                                    end = start + len(all_i[-1].text)
                                elif elem.c:
                                    all_c = elem.find_all("c")
                                    item["text"] += all_c[-1].text
                                    end = start + len(all_c[-1].text)
                                item["other_ided_ns"].append({"start":start, "end":end})
                            else:
                                item["text"] += elem.text
                        yield item
            elif member.name.endswith("jsonl") or member.name.endswith("json"):
                for line in ifd:
                    j = json.loads(line)
                    item = {
                        "text" : j["sample"],
                        "annotations" : [],
                        "other_ided_ns": []
                    }
                    for obs, props in j["words"].items():
                        for start in props["i"]:
                            if props["Std"] != "{}":
                                end = start + len(obs)
                                item["annotations"].append(
                                    {
                                        "start" : start,
                                        "end" : end,
                                        "observed" : item["text"][start:end],
                                        "standard" : props["Std"],
                                        "ocr" : props.get("OCR", False)
                                    }
                                )
                    yield item


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", dest="input_file",  help="Input file")
    parser.add_argument("--output_file", dest="output_file",  help="Output file")
    parser.add_argument("--max_ld", dest="max_ld", type=int, default=3,  help="Max LD")
    parser.add_argument("--custom_ld", type=bool, default=False, help="use weighted LD")
    args, rest = parser.parse_known_args()

    #brown_wn_tree = loadBrownWNCorpusTree()
    brown_tree = loadBrownCorpusTree() if not args.custom_ld else loadBrownCustomTree()
    #sw_tree = loadSWTree()

    with gzip.open(args.output_file, "wt") as ofd:

        for item in loadCorpus(args.input_file):
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
                                #"alts": {a[1]: a[0] for a in brown_wn_tree.find(annotation["observed"].lower(), args.max_ld)}
                                "alts": {a[1]: a[0] for a in brown_tree.find(annotation["observed"].lower(), args.max_ld) if len(wn.synsets(a[1])) > 0 or a[1] in stopwords.words("english")}                     

                            }
                        )
                offset = new_offset
                if len(sentence_item["annotations"]) > 0:
                    ofd.write(json.dumps(sentence_item) + "\n")
