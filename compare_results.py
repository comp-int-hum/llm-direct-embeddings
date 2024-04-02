import pandas as pd
import argparse

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--csv_1", help="Detailed results CSV")
        parser.add_argument("--csv_2", help="Detailed results CSV")
        args, rest = parser.parse_known_args()

        df_one = pd.read_csv(args.csv_1, index_col="observed")
        df_two = pd.read_csv(args.csv_2, index_col="observed")

        print(df_one)
        print(df_two)

        print("Found in" + args.csv_1 + " and not in comparison")
        print(" ".join(df_one.index.difference(df_two.index).tolist()))
        print("Found in" + args.csv_2 + " and not in comparison")
        print(" ".join(df_two.index.difference(df_one.index).tolist()))
