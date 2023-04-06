import argparse
import json
import gzip
import math


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input file")
    parser.add_argument("output_files", default=[], nargs="+",  help="Output files")
    args, rest = parser.parse_known_args()

    items = []
    with gzip.open(args.input_file, "rt") as ifd:
        for line in ifd:
            items.append(line)

    total = len(items)
    num_chunks = len(args.output_files)
    max_per_chunk = math.ceil(total / num_chunks)

    for i, fname in enumerate(args.output_files):
        with gzip.open(fname, "wt") as ofd:
            for line in items[i * max_per_chunk : (i + 1) * max_per_chunk]:
                ofd.write(line)

