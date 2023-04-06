import argparse
import re
import gzip
import json
import tarfile
from bs4 import BeautifulSoup


def loadCorpus(fname, sent_sep=True):
    with tarfile.open(fname, "r") as tifd:
        for member in [m for m in tifd.getmembers() if m.isfile()]:
            ifd = tifd.extractfile(member)
            if member.name.endswith("xml"):
                soup = BeautifulSoup(ifd, "xml")
                for ca in soup.find_all("coded_answer"):
                    for p in ca.find_all("p"):
                        item = {
                            "text" : "",
                            "annotations" : []
                        }
                        for elem in p:
                            if elem.name == "NS" and elem.i and elem.c and elem["type"] in ["S", "SA", "SX"]:
                                start = len(item["text"])
                                end = start + len(elem.i.text)
                                item["text"] += elem.i.text
                                item["annotations"].append(
                                    {
                                        "start" : start,
                                        "end" : end,
                                        "standard" : elem.c.text,
                                        "observed" : elem.i.text,
                                        "ocr" : False
                                    }
                                )
                            else:
                                item["text"] += elem.text
                        yield item
            elif member.name.endswith("jsonl"):
                for line in ifd:
                    j = json.loads(line)
                    item = {
                        "text" : j["sample"],
                        "annotations" : []
                    }
                    for obs, props in j["words"].items():
                        for start in props["i"]:
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
    args, rest = parser.parse_known_args()
    
    with gzip.open(args.output_file, "wt") as ofd:
        for item in loadCorpus(args.input_file):
            offset = 0
            for sentence_match in re.finditer(r"(.+?(?:(?:(?:\.|\?|\!)\s+)|$))", item["text"], re.S):
                sentence = sentence_match.group(1)
                sentence_item = {
                    "text" : sentence,
                    "annotations" : []
                }
                new_offset = offset + len(sentence)
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
                                "observed" : annotation["observed"],
                                "standard" : annotation["standard"],
                                "ocr" : annotation["ocr"]                                
                            }
                        )
                offset = new_offset
                if len(sentence_item["annotations"]) > 0:
                    ofd.write(json.dumps(sentence_item) + "\n")
