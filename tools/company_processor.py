import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from tqdm import tqdm
import pickle

def main(args):

    if os.path.exists(os.path.join(args.company_dir, f"index_{args.index}", "company_dict.pkl")):
        sys.exit(f"Company data already exists. Aborting.")

    train_mapping_df = pd.read_csv(os.path.join(args.data_dir, "map_date_train.csv"), parse_dates=[1])
    test_mapping_df = pd.read_csv(os.path.join(args.data_dir, "map_date_test.csv"), parse_dates=[1])

    company_dict = {}
    company_embeddings_dir = os.path.join(args.company_dir)
    for filename in tqdm(os.listdir(company_embeddings_dir), desc=f"Processing company dict: index {args.index}"):
        if filename.endswith(".npy"):
            u3_id = os.path.splitext(filename)[0]
            embedding = np.load(os.path.join(company_embeddings_dir, filename))
            company_dict[u3_id] = embedding
    
    with open(os.path.join(args.company_dir, f"index_{args.index}", "company_dict.pkl"), "wb") as f:
        pickle.dump(company_dict, f)

    test_company_keys = []
    for id in tqdm(test_mapping_df["company_id"], desc=f"Processing test company ids: index {args.index}"):
        test_company_keys.append(str(int(id)))
    test_company_keys = np.array(test_company_keys)

    train_company_keys = []
    for id in tqdm(train_mapping_df["company_id"], desc=f"Processing train company ids: index {args.index}"):
        train_company_keys.append(str(int(id)))
    train_company_keys = np.array(train_company_keys)

    np.save(os.path.join(args.company_dir, f"index_{args.index}", "test_company_keys.npy"), test_company_keys)
    np.save(os.path.join(args.company_dir, f"index_{args.index}", "train_company_keys.npy"), train_company_keys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--company_dir",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
    )
    parser.add_argument(
        "--index",
        type=int,
        default=1
    )
 
    args = parser.parse_args()
    if (args.index != int(args.data_dir[-1])):
        raise ValueError("Index argument does not match data_dir index.")
    main(args)