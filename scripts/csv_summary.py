import argparse
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", help="Individual source summaries")
    parser.add_argument("--summary_out", help="Final summary out")

    args, rest = parser.parse_known_args()

    with open(args.summary_out, "wt", newline="") as sout:
        sowriter = csv.writer(sout)
        sowriter.writerow(["Model", "Max LD","N", "n_Acc", "n_Inacc", "Accuracy", "In alts %", "Inacc not in alts", "Acc alt present", "Acc 1ns", "N 1ns"
                           ,"Stemmed acc", "Stemmed acc alt present", "Acc only model"])
        for source in args.sources:
            with open(source, "rt") as sin:
                s_reader = csv.DictReader(sin)
                for row in s_reader:
                    sowriter.writerow([v for v in row.values()])
                    
        
