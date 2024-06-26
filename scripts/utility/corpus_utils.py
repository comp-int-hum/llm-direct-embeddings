import pandas as pd
import json
import glob
import re

from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import pybktree
import jellyfish

from collections import Counter

from bs4 import BeautifulSoup
from .ld import custom_ld

def loadJsonCorpusDf(corpus_in, sent_sep=True):
	initial_res = []
	for sample in corpus_in: #seperate load function to get NS, Ground OCR into DF, then can apply
		js_sample = json.loads(sample)
		js_sample["sample"] = js_sample["sample"].lower()
		for word, entry in js_sample["words"].items():
			for i in entry["i"]:
				if sent_sep:
					#convoluted approach to getting new index
					sep = re.split("[!.?]", js_sample["sample"])
					new_i = i
					for sublist in sep:
						if new_i < (len(sublist)+1):
							initial_res.append({"sample": sublist, "i": new_i, "NS": word, "Ground": entry["Std"], "OCR": int(entry["OCR"])})
							break
						new_i -= (len(sublist)+1)


				else:
					initial_res.append({"sample": js_sample["sample"], "i": i, "NS": word, "Ground": entry["Std"], "OCR": int(entry["OCR"])})
	return pd.DataFrame.from_records(initial_res)

def loadFCECorpusDf(dataset, allowed_error_types = ["S","SA","SX"]):
	initial_res = []
	for corpus_file in dataset:
		print(corpus_file)
		with open(corpus_file) as x_in:
			soup = BeautifulSoup(x_in, "xml")
			for ca in soup.find_all("coded_answer"):
				for p in ca.find_all("p"):
					include=False
					p_str=""
					for item in p:
						if item.name == "NS":
							if item["type"] in allowed_error_types:
								try:
									include=True
									p_str+="|"+item.i.text+"|"+item.c.text+"|"
								except AttributeError:
									pass
							else:
								if item.c:
									p_str+=item.c.text
								elif item.i:
									p_str+=item.i.text
						else:
							p_str+=item.text
					if include:
						#split on punct
						for sent in re.split("[!.?]", p_str):
							#then split on the bars,
							bar_split = sent.lower().split("|")
							print(bar_split)
							#to replicate, throw out sent with > 1 spelling error
							if len(bar_split) == 4:
								initial_res.append({"sample": bar_split[0]+bar_split[1]+bar_split[3], "i": len(bar_split[0]), "NS": bar_split[1], "Ground": bar_split[2], "OCR": int(False)})

	return pd.DataFrame.from_records(initial_res)


def maskSample(text, annotation, mask_token = "[MASK]"):
        masked_sample = text[0:annotation["start"]] + mask_token +  text[annotation["end"]:]
        return masked_sample


def loadBrownCorpusTree():
	corpus_dictionary = Counter([word.lower() for word in brown.words() if word.isalpha() and not word[0].isupper()])
	corpus_dictionary = [w for w,c in corpus_dictionary.items() if c > 0]
	return pybktree.BKTree(jellyfish.levenshtein_distance, corpus_dictionary)

def loadBrownWNCorpusTree():
	corpus_dictionary = Counter([word.lower() for word in brown.words() if word.isalpha() and not word[0].isupper()])
	corpus_dictionary = [w for w,c in corpus_dictionary.items() if c > 1]
	corpus_dictionary = list(set(corpus_dictionary).union(set([word for word in wn.words() if "_" not in word])))
	return pybktree.BKTree(jellyfish.levenshtein_distance, corpus_dictionary)

def loadSWTree():
	return pybktree.BKTree(jellyfish.levenshtein_distance, [w for w in stopwords("english")])

def loadBrownCustomTree():
	corpus_dictionary = Counter([word.lower() for word in brown.words() if word.isalpha() and not word[0].isupper()])
	corpus_dictionary = [w for w,c in corpus_dictionary.items() if c > 0]
	return pybktree.BKTree(custom_ld.levenshtein, corpus_dictionary)

def genOCRError(orig):
        if "s" in orig:
                return orig.replace("s", "5", 1)
        if "o" in orig:
                return orig.replace("o", "0", 1)
        if "n" in orig:
                return orig.replace("n", "h", 1)
        orig = "l"+orig[1:]
        return orig


