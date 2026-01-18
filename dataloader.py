import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from utils.masking import mask_target
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def custom_collate(batch):
    x, company_embedding, macro_seq, y = zip(*batch)
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    company_embedding = torch.stack(company_embedding, dim=0)

    max_chunks = max(t.shape[0] for s in macro_seq for t in s)
    D = 1536

    # pad within sample
    samples = []
    chunk_lens = []
    for s in macro_seq:
        chunk_lens.append([t.shape[0] for t in s])
        if len(s) == 0:
            samples.append(torch.zeros(0, max_chunks, D))
            continue
        padded_s = pad_sequence(s, batch_first=True, padding_value=0)
        if (padded_s.size(1) < max_chunks):
            extra = max_chunks - padded_s.size(1)
            padded_s = torch.nn.functional.pad(padded_s, (0, 0, 0, extra), "constant", 0)
        samples.append(padded_s)
    
    # pad across samples
    macro_embeddings = pad_sequence(samples, batch_first=True, padding_value=0)

    # mask
    B, T, C, D = macro_embeddings.shape
    chunk_masks = torch.zeros((B, T, C), dtype=torch.bool)
    for b, chunk_len in enumerate(chunk_lens):
        for c, l in enumerate(chunk_len):
            chunk_masks[b, c, :l] = 1
    macro_masks = chunk_masks

    return (x, company_embedding, macro_embeddings, macro_masks, y)


def load_raw_data(
    data_dir, level=0
):
    train_input = np.load(os.path.join(data_dir, "x_train.npy"))
    train_macro_keys = np.load(os.path.join(data_dir, "train_macro_keys.npy"), allow_pickle=True)
    train_company_keys = np.load(os.path.join(data_dir, "train_company_keys.npy"), allow_pickle=True)
    train_target = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    train_mapping_df = pd.read_csv(os.path.join(data_dir, "map_date_train.csv"), parse_dates=[1])
    train_mapping_df["company_id"] = train_mapping_df["company_id"].astype(int)
    cum_label_train = pd.read_csv(os.path.join(data_dir, "y_cum_train.csv"))
    cum_label_train = cum_label_train.replace({2:0})
    test_input = np.load(os.path.join(data_dir, "x_test.npy"))
    test_macro_keys = np.load(os.path.join(data_dir, "test_macro_keys.npy"), allow_pickle=True)
    test_company_keys = np.load(os.path.join(data_dir, "test_company_keys.npy"), allow_pickle=True)
    test_target = pd.read_csv(os.path.join(data_dir, "y_test.csv"))
    test_mapping_df = pd.read_csv(os.path.join(data_dir, "map_date_test.csv"), parse_dates=[1])
    test_mapping_df["company_id"] = test_mapping_df["company_id"].astype(int)
    cum_label_test = pd.read_csv(os.path.join(data_dir, "y_cum_test.csv"))
    cum_label_test = cum_label_test.replace({2:0})
    with open(os.path.join(data_dir, "fomc_dict.pkl"), "rb") as f:
        macro_dict = pd.read_pickle(f)
    
    if level == 4:
        with open(os.path.join(data_dir, "company_dict_2.pkl"), "rb") as f:
            company_dict = pd.read_pickle(f)
    else:
        with open(os.path.join(data_dir, "company_dict_1.pkl"), "rb") as f:
            company_dict = pd.read_pickle(f)

    return train_input, train_company_keys, train_macro_keys, train_target, train_mapping_df, cum_label_train, \
        test_input, test_company_keys, test_macro_keys, test_target, test_mapping_df, cum_label_test, \
        company_dict, macro_dict


class Dataset(Dataset):
    def __init__(self, input, target, company_keys, company_dict, macro_keys, macro_dict, use_macro=False, n_macro=1, level=0):
        self.input = torch.as_tensor(input, dtype=torch.float32)
        self.target = torch.as_tensor(target.values, dtype=torch.long)
        if use_macro:
            self.n_macro = n_macro
            self.macro_keys = macro_keys
            self.macro_dict = macro_dict
            self.company_keys = company_keys
            self.company_dict = company_dict
        self.use_macro = use_macro
        self.level = level

    def __getitem__(self, index):
        x = self.input[index]
        y = self.target[index]
        if not self.use_macro:
            return (x, y)
        company_key = self.company_keys[index]
        company_embedding = torch.as_tensor(self.company_dict[company_key], dtype=torch.float32)
        macro_keys = sorted(self.macro_keys[index])[-self.n_macro:]
        macro_seq = [torch.as_tensor(self.macro_dict[k], dtype=torch.float32) for k in macro_keys]
        if self.level == 1:
            macro_seq = [torch.zeros_like(macro) for macro in macro_seq] 
        elif self.level == 2:
            company_embedding = torch.zeros_like(company_embedding)
        return (x, company_embedding, macro_seq, y)

    def __len__(self):
        return len(self.target)


def prepare_labeled_loader(
    train_input,
    train_target,
    test_input,
    test_target,
    train_company_keys=None,
    test_company_keys=None,
    company_dict=None,
    train_macro_keys=None,
    test_macro_keys=None,
    macro_dict=None,
    **dataloader_kwargs,
):
    batch_size = dataloader_kwargs.get("batch_size", 1024)
    num_workers = dataloader_kwargs.get("num_workers", 10)
    pin_memory = dataloader_kwargs.get("pin_memory", True)
    shuffle = dataloader_kwargs.get("shuffle", True)
    use_macro = dataloader_kwargs.get("use_macro", False)
    n_macro = dataloader_kwargs.get("n_macro", 1)
    level = dataloader_kwargs.get("level", 0)
    print("Level:", level)

    train_dataset = Dataset(
        train_input,
        train_target,
        train_company_keys, 
        company_dict,
        train_macro_keys,
        macro_dict,
        use_macro,
        n_macro,
        level
    )
    test_dataset = Dataset(
        test_input,
        test_target,
        test_company_keys, 
        company_dict,
        test_macro_keys,
        macro_dict, 
        use_macro,
        n_macro,
        level
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate if use_macro else None,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate if use_macro else None,
    )

    return train_dataloader, test_dataloader


def prepare_inference_loader(
    input,
    target,
    company_keys=None,
    company_dict=None,
    macro_keys=None,
    macro_dict=None,
    **dataloader_kwargs,
):
    batch_size = dataloader_kwargs.get("batch_size", 1024)
    num_workers = dataloader_kwargs.get("num_workers", 10)
    pin_memory = dataloader_kwargs.get("pin_memory", True)
    use_macro = dataloader_kwargs.get("use_macro", False)
    n_macro = dataloader_kwargs.get("n_macro", 1)
    level = dataloader_kwargs.get("level", 0)
    print("Level:", level)

    dataset = Dataset(
        input,
        target,
        company_keys,
        company_dict,
        macro_keys,
        macro_dict,
        use_macro,
        n_macro,
        level
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate if use_macro else None,
    )

    return dataloader


def prepare_cross_sectional_labeled_loader(
    train_input,
    train_target,
    train_mapping_df,
    seed,
    company_keys=None,
    company_dict=None,
    macro_keys=None,
    macro_dict=None,
    **dataloader_kwargs,
):
    use_macro = dataloader_kwargs.get("use_macro", False)
    _train_df, _test_df = train_test_split(train_mapping_df, test_size=0.2, random_state=seed)
    train_idxs = _train_df.index
    test_idxs = _test_df.index
    _train_input = train_input[train_idxs]
    _train_target = train_target.iloc[train_idxs].reset_index(drop=True)
    _test_input = train_input[test_idxs]
    _test_target = train_target.iloc[test_idxs].reset_index(drop=True)
    if use_macro:
        _train_company_keys = company_keys[train_idxs]
        _test_company_keys = company_keys[test_idxs]
        _train_macro_keys = macro_keys[train_idxs]
        _test_macro_keys = macro_keys[test_idxs]
    else:
        _train_company_keys = None
        _test_company_keys = None
        _train_macro_keys = None
        _test_macro_keys = None

    return prepare_labeled_loader(_train_input, _train_target, _test_input, _test_target, _train_company_keys, _test_company_keys, company_dict, _train_macro_keys, _test_macro_keys, macro_dict, **dataloader_kwargs)


def prepare_cross_sectional_inference_loader(
    test_input,
    test_target,
    test_company_keys=None,
    company_dict=None,
    test_macro_keys=None,
    macro_dict=None,
    **dataloader_kwargs,
):
    return prepare_inference_loader(test_input, test_target, test_company_keys, company_dict, test_macro_keys, macro_dict, **dataloader_kwargs)


def prepare_overtime_rolling_labeled_loader(
    train_input,
    train_target,
    test_input,
    test_target,
    train_mapping_df,
    test_mapping_df,
    sample_date,
    horizon_size,
    train_company_keys=None,
    test_company_keys=None,
    company_dict=None,
    train_macro_keys=None,
    test_macro_keys=None,
    macro_dict=None,
    **dataloader_kwargs,
):
    use_macro = dataloader_kwargs.get("use_macro", False)

    start = sample_date - pd.DateOffset(months=120)
    cut = sample_date - pd.DateOffset(months=24)
    end = sample_date

    input = np.concatenate([train_input, test_input])
    target = pd.concat([train_target, test_target], ignore_index=True)
    mapping_df = pd.concat([train_mapping_df, test_mapping_df], ignore_index=True)
    mapping_df = mapping_df[
        (mapping_df["date"].dt.year >= start.year) &
        (mapping_df["date"].dt.year < end.year)
    ]
    print(f"Training data from {(sample_date - pd.DateOffset(months=120)).year} to {sample_date.year - 1}")

    train_mask = (mapping_df["date"].dt.year >= start.year) & (mapping_df["date"].dt.year < cut.year)
    test_mask = (mapping_df["date"].dt.year >= cut.year) & (mapping_df["date"].dt.year < end.year)
    train_idxs = mapping_df.index[train_mask]
    test_idxs = mapping_df.index[test_mask]
   
    _train_df = mapping_df[train_mask]
    _test_df = mapping_df[test_mask]
    _train_input = input[train_idxs]
    _test_input = input[test_idxs]
    _train_target = target.iloc[train_idxs].reset_index(drop=True)
    _test_target = target.iloc[test_idxs].reset_index(drop=True)

    if use_macro:
        company_keys = np.concatenate([train_company_keys, test_company_keys])
        _train_company_keys = company_keys[train_idxs]
        _test_company_keys = company_keys[test_idxs]
        macro_keys = np.concatenate([train_macro_keys, test_macro_keys])
        _train_macro_keys = macro_keys[train_idxs]
        _test_macro_keys = macro_keys[test_idxs]
    else:
        _train_company_keys = None
        _test_company_keys = None
        _train_macro_keys = None
        _test_macro_keys = None

    mask_target(_train_target, _test_target, _train_df, _test_df, sample_date, horizon_size)

    return prepare_labeled_loader(_train_input, _train_target, _test_input, _test_target, _train_company_keys, _test_company_keys, company_dict, _train_macro_keys, _test_macro_keys, macro_dict, **dataloader_kwargs)


def prepare_overtime_expanding_labeled_loader(
    train_input,
    train_target,
    test_input,
    test_target,
    train_mapping_df,
    test_mapping_df,
    sample_date,
    horizon_size,
    train_company_keys=None,
    test_company_keys=None,
    company_dict=None,
    train_macro_keys=None,
    test_macro_keys=None,
    macro_dict=None,
    **dataloader_kwargs,
):
    use_macro = dataloader_kwargs.get("use_macro", False)

    cut = sample_date - pd.DateOffset(months=24)
    end = sample_date

    input = np.concatenate([train_input, test_input])
    target = pd.concat([train_target, test_target], ignore_index=True)
    mapping_df = pd.concat([train_mapping_df, test_mapping_df], ignore_index=True)
    mapping_df = mapping_df[
        (mapping_df["date"].dt.year < end.year)
    ]
    print(f"Training data from earlist date to {sample_date.year - 1}")

    train_mask = (mapping_df["date"].dt.year < cut.year)
    test_mask = (mapping_df["date"].dt.year >= cut.year) & (mapping_df["date"].dt.year < end.year)
    train_idxs = mapping_df.index[train_mask]
    test_idxs = mapping_df.index[test_mask]
   
    _train_df = mapping_df[train_mask]
    _test_df = mapping_df[test_mask]
    _train_input = input[train_idxs]
    _test_input = input[test_idxs]
    _train_target = target.iloc[train_idxs].reset_index(drop=True)
    _test_target = target.iloc[test_idxs].reset_index(drop=True)

    if use_macro:
        company_keys = np.concatenate([train_company_keys, test_company_keys])
        _train_company_keys = company_keys[train_idxs]
        _test_company_keys = company_keys[test_idxs]
        macro_keys = np.concatenate([train_macro_keys, test_macro_keys])
        _train_macro_keys = macro_keys[train_idxs]
        _test_macro_keys = macro_keys[test_idxs]
    else:
        _train_company_keys = None
        _test_company_keys = None
        _train_macro_keys = None
        _test_macro_keys = None

    mask_target(_train_target, _test_target, _train_df, _test_df, sample_date, horizon_size)

    return prepare_labeled_loader(_train_input, _train_target, _test_input, _test_target, _train_company_keys, _test_company_keys, company_dict, _train_macro_keys, _test_macro_keys, macro_dict, **dataloader_kwargs)


def prepare_overtime_inference_loader(
    train_input,
    train_target,
    test_input,
    test_target,
    train_mapping_df,
    test_mapping_df,
    sample_date,
    train_company_keys=None,
    test_company_keys=None,
    company_dict=None,
    train_macro_keys=None,
    test_macro_keys=None,
    macro_dict=None,
    company_id=None,
    **dataloader_kwargs,
):
    use_macro = dataloader_kwargs.get("use_macro", False)
    if company_id:
        sample_train_mapping_df = train_mapping_df[
            (train_mapping_df["date"].dt.year == sample_date.year) & 
            (train_mapping_df["date"].dt.month == sample_date.month) &
            (train_mapping_df["company_id"] == company_id)
        ]
        sample_test_mapping_df = test_mapping_df[
            (test_mapping_df["date"].dt.year == sample_date.year) & 
            (test_mapping_df["date"].dt.month == sample_date.month) &
            (test_mapping_df["company_id"] == company_id)
        ]
    else:
        sample_train_mapping_df = train_mapping_df[
            (train_mapping_df["date"].dt.year == sample_date.year) & 
            (train_mapping_df["date"].dt.month == sample_date.month)
        ]
        sample_test_mapping_df = test_mapping_df[
            (test_mapping_df["date"].dt.year == sample_date.year) & 
            (test_mapping_df["date"].dt.month == sample_date.month)
        ]
    sample_train_idxs = sample_train_mapping_df.index
    sample_test_idxs = sample_test_mapping_df.index
    input = np.concatenate([train_input[sample_train_idxs], test_input[sample_test_idxs]])
    target = pd.concat([train_target.iloc[sample_train_idxs], test_target.iloc[sample_test_idxs]], ignore_index=True)
    macro_keys = None
    company_keys = None
    if use_macro:
        macro_keys = np.concatenate([train_macro_keys[sample_train_idxs], test_macro_keys[sample_test_idxs]])
        company_keys = np.concatenate([train_company_keys[sample_train_idxs], test_company_keys[sample_test_idxs]])
    return prepare_inference_loader(input, target, company_keys, company_dict, macro_keys, macro_dict, **dataloader_kwargs)


def prepare_overtime_eval_data(
    train_mapping_df,
    test_mapping_df,
    cum_label_train,
    cum_label_test,
    sample_date,
    company_id=None
):
    if company_id:
        sample_train_mapping_df = train_mapping_df[
            (train_mapping_df["date"].dt.year == sample_date.year) & 
            (train_mapping_df["date"].dt.month == sample_date.month) &
            (train_mapping_df["company_id"] == company_id)
        ]
        sample_test_mapping_df = test_mapping_df[
            (test_mapping_df["date"].dt.year == sample_date.year) & 
            (test_mapping_df["date"].dt.month == sample_date.month) &
            (test_mapping_df["company_id"] == company_id)
        ]
    else:
        sample_train_mapping_df = train_mapping_df[
            (train_mapping_df["date"].dt.year == sample_date.year) & 
            (train_mapping_df["date"].dt.month == sample_date.month)
        ]
        sample_test_mapping_df = test_mapping_df[
            (test_mapping_df["date"].dt.year == sample_date.year) & 
            (test_mapping_df["date"].dt.month == sample_date.month)
        ]
    sample_train_idxs = sample_train_mapping_df.index
    sample_test_idxs = sample_test_mapping_df.index
    mapping_df = pd.concat([sample_train_mapping_df, sample_test_mapping_df], ignore_index=True)
    cum_label = pd.concat([cum_label_train.iloc[sample_train_idxs], cum_label_test.iloc[sample_test_idxs]], ignore_index=True)
    return mapping_df, cum_label.values