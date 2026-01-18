import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import sys
from tqdm import tqdm
import pickle

def main(args):

    train_mapping_df = pd.read_csv(os.path.join(args.data_dir, "map_date_train.csv"), parse_dates=[1])
    test_mapping_df = pd.read_csv(os.path.join(args.data_dir, "map_date_test.csv"), parse_dates=[1])

    fomc_dict = {}
    fomc_embeddings_dir = os.path.join(args.fomc_dir)
    for filename in os.listdir(fomc_embeddings_dir):
        if filename.endswith(".npy"):
            date = os.path.splitext(filename)[0]
            dt = datetime.strptime(str(date), "%Y%m%d")
            embedding = np.load(os.path.join(fomc_embeddings_dir, filename))
            fomc_dict[dt] = embedding
    with open(os.path.join(args.fomc_dir, f"index_{args.index}", "fomc_dict.pkl"), "wb") as f:
        pickle.dump(fomc_dict, f)

    test_macro_keys = []
    for index, row in tqdm(test_mapping_df.iterrows(), total = test_mapping_df.shape[0], desc=f"Processing test dates: index {args.index}"):
        end_date = row["date"]
        start_date = end_date - relativedelta(months=12)
        eligible_keys = [k for k in fomc_dict.keys() if k and start_date <= k <= end_date]
        test_macro_keys.append(eligible_keys)
    test_macro_keys = np.array(test_macro_keys, dtype=object)

    train_macro_keys = []
    for index, row in tqdm(train_mapping_df.iterrows(), total = train_mapping_df.shape[0], desc=f"Processing train dates: index {args.index}"):
        end_date = row["date"]
        start_date = end_date - relativedelta(months=12)
        eligible_keys = [k for k in fomc_dict.keys() if k and start_date <= k <= end_date]
        train_macro_keys.append(eligible_keys)
    train_macro_keys = np.array(train_macro_keys, dtype=object)
    
    np.save(os.path.join(args.fomc_dir, f"index_{args.index}", "test_macro_keys.npy"), test_macro_keys)
    np.save(os.path.join(args.fomc_dir, f"index_{args.index}", "train_macro_keys.npy"), train_macro_keys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fomc_dir",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0
    )
 
    args = parser.parse_args()
    if (args.index != int(args.data_dir[-1])):
        raise ValueError("Index argument does not match data_dir index.")
    main(args)