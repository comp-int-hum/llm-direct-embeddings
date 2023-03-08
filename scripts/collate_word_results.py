import argparse



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("individual_results", nargs="+", help="A list of csv files containing individual results")
	parser.add_argument("--layers", nargs="+", default=["-1"], dest="layers")

	args, rest = parser.parse_known_args()