import sys
sys.path.append("..")
import os
import sys
import numpy as np
import argparse
import torch
from tqdm import tqdm
from dataloader import load_raw_data, prepare_overtime_inference_loader, prepare_overtime_eval_data
from utils.setup_model import setup_model
from utils.configs import get_dataloader_kwargs
from utils.generator import experiment_name
from utils.probability import get_foward_probability, get_cumulative_probability
import pandas as pd
from matplotlib import pyplot as plt
import pickle

def eval(args, inference_date, data):

    # Set up experiment name
    exp_name = experiment_name(args)
    year = inference_date.year
    exp_name = f"{exp_name}_{year}"
    print(f"Evaluatoin: [{inference_date}, {exp_name}]")

    # Set up device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Set up model
    model_dir = os.path.join("runs", exp_name)
    with open(f"{model_dir}/args.pkl", "rb") as f:
        model_configs = pickle.load(f)
    model = setup_model(model_configs)
    para = torch.load(f"{model_dir}/best_ckpt.pt", weights_only=False)
    model.load_state_dict(para["model_state_dict"])
    model.to(device)

    # Load data
    train_input, train_company_keys, train_macro_keys, train_target, train_mapping_df, cum_label_train, \
    test_input, test_company_keys, test_macro_keys, test_target, test_mapping_df, cum_label_test, \
    company_dict, macro_dict = load_raw_data(model_configs.data_dir)

    dataloader_kwargs = get_dataloader_kwargs(model_configs)
    if model_configs.use_macro:
        test_data = prepare_overtime_inference_loader(
            train_input, train_target, test_input, test_target, train_mapping_df, test_mapping_df, inference_date, \
            train_company_keys, test_company_keys, company_dict, train_macro_keys, test_macro_keys, macro_dict, **dataloader_kwargs
        )
    else:
        test_data = prepare_overtime_inference_loader(
            train_input, train_target, test_input, test_target, train_mapping_df, test_mapping_df, inference_date, **dataloader_kwargs
        )
    mapping_df, cum_label = prepare_overtime_eval_data(
        train_mapping_df, test_mapping_df, cum_label_train, cum_label_test, inference_date
    )

    # Inference
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_data, desc=f"Overtime Inference", leave=False, ncols=80):
            *inputs, _ = batch
            inputs = [input.to(device) for input in inputs]
            if model_configs.use_macro:
                x, c, m, m_mask = inputs
                y_pred = model(x, c, m, m_mask)
            else:
                x = inputs[0]
                y_pred = model(x)
            predictions.extend(y_pred.cpu().numpy())
        predictions = np.array(predictions, dtype=np.float32)
    
    forward_default_probability = get_foward_probability(
        predictions=predictions,
        max_horizon=model_configs.horizon_size,
        CRI=True
    )

    cumulative_default_probability = get_cumulative_probability(
        forward_default_probability
    )

    dates = mapping_df.iloc[:, 1].values
    com_num = mapping_df.groupby('date').size().to_frame('size')

    horizon_list = [0, 2, 5, 11, 23, 35]
    predict_list = []
    observed_list = []

    for horizon in horizon_list:

        mask_invalid = (cum_label[:,horizon] != -1)
        final_mask = mask_invalid

        # Dataframe with 3 columns (dates, labels, and predictions)
        calc_data = pd.DataFrame(dates[final_mask], columns=['date'])
        calc_data['{}M_l'.format(horizon + 1)] = cum_label[final_mask, horizon]
        calc_data['{}M_p'.format(horizon + 1)] = cumulative_default_probability[final_mask, horizon]
        if calc_data.shape[0] != 0:
            calc_data_g = calc_data.groupby('date').sum()
            calc_data_g = pd.concat([calc_data_g,com_num], axis=1, join='inner')
            predict_list.append(calc_data_g['{}M_p'.format(horizon + 1)].values[0])
            observed_list.append(calc_data_g['{}M_l'.format(horizon + 1)].values[0])

    return predict_list, observed_list

def main(args):
    # Load data
    data =  load_raw_data(args.data_dir)
    _, _, _, _, train_mapping_df, _,  \
    _, _, _, _, test_mapping_df, _, \
    _, _ = data

    earliest_date = min(train_mapping_df["date"].min(), test_mapping_df["date"].min())
    inference_date = earliest_date + pd.DateOffset(months = args.init_overtime_window)
    horizon_list = [0, 2, 5, 11, 23, 35]
    results = []
    
    while True:
        predict_list, observed_list = eval(args, inference_date, data)

        row = {"date": inference_date}
        for i, val in enumerate(predict_list):
            row[f"predict_{horizon_list[i] + 1}"] = val
        for i, val in enumerate(observed_list):
            row[f"observed_{horizon_list[i] + 1}"] = val

        results.append(row)

        inference_date += pd.DateOffset(months = 1)
        if inference_date > train_mapping_df["date"].max() - pd.DateOffset(months = 36) \
            and inference_date > test_mapping_df["date"].max() - pd.DateOffset(months = 36):
            break

    result_df = pd.DataFrame(results)
    result_df["date"] = pd.to_datetime(result_df["date"])
    result_df = result_df.set_index("date")

    result_df.to_csv(f"experiments/{args.exp_name}_overtime_cnt.csv")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for i, h in enumerate(horizon_list):
        ax = axes[i]
        ax.plot(result_df.index.to_numpy(), result_df[f"predict_{h + 1}"].values, color="red", linewidth=1, label="Predict")
        ax.bar(result_df.index.to_numpy(), result_df[f"observed_{h + 1}"].values, width=1, align='center', color="#00A2FF", edgecolor="#00A2FF",alpha=0.7, label="Observed")
        ax.set_title(f"{h + 1} m")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"experiments/{args.exp_name}_count.png")
    print("Count plot saved to experiments/{args.exp_name}_count.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment parameters
    parser.add_argument(
        "--exp_name", required=True, type=str, default=None, help="Name of experiment"
    )
    parser.add_argument(
        "--data_dir", required=True, type=str, default="new_data", help="Directory containing the dataset"
    )
    parser.add_argument(
        "--init_overtime_window", type=int, default=120, help="Initial months of overtime training"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=2, help="The specific number of gpu"
    )

    args = parser.parse_args()

    # Print Config
    print(args)

    main(args)