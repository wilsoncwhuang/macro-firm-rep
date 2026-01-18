import pandas as pd
import torch

# Mask all target value after sample date 
def mask_target(train_target, test_target, train_mapping_df, test_mapping_df, sample_date, horizon_size):
    train_target = train_target.copy()
    test_target = test_target.copy()
    mask_start_date = sample_date - pd.DateOffset(months=horizon_size + 1)
    selected_train_mask_cnt_df = ((train_mapping_df["date"].dt.year - mask_start_date.year) * 12 + (train_mapping_df["date"].dt.month - mask_start_date.month)).reset_index(drop=True)
    selected_test_mask_cnt_df = ((test_mapping_df["date"].dt.year - mask_start_date.year) * 12 + (test_mapping_df["date"].dt.month - mask_start_date.month)).reset_index(drop=True)
    
    for i, n in selected_train_mask_cnt_df.items():
        if n > 0:
            train_target.iloc[i, -n:] = -1
    for i, n in selected_test_mask_cnt_df.items():
        if n > 0:
            test_target.iloc[i, -n:] = -1
    return train_target, test_target

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask