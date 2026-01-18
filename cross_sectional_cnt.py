import sys
sys.path.append("..")
import os
import sys
import numpy as np
import argparse
import torch
from tqdm import tqdm
from dataloader import load_raw_data, prepare_cross_sectional_inference_loader
from utils.setup_model import setup_model
from utils.configs import get_dataloader_kwargs
from utils.generator import experiment_name
from utils.probability import get_foward_probability, get_cumulative_probability
import pandas as pd
from matplotlib import pyplot as plt
import pickle

def eval(args, run):

    # Set up experiment name
    exp_name = experiment_name(args)
    exp_name = f"{exp_name}_{run}"
    print(f"Evaluatoin: {exp_name}")

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

    # Load Data
    train_input, train_company_keys, train_macro_keys, train_target, train_mapping_df, cum_label_train, \
    test_input, test_company_keys, test_macro_keys, test_target, test_mapping_df, cum_label_test, \
    company_dict, macro_dict = load_raw_data(model_configs.data_dir)
    dataloader_kwargs = get_dataloader_kwargs(model_configs)
    if args.inference_type == "train":
        input = train_input
        company_keys = train_company_keys
        macro_keys = train_macro_keys
        target = train_target
        mapping_df = train_mapping_df
        cum_label = cum_label_train.values
        if model_configs.use_macro:
            test_data = prepare_cross_sectional_inference_loader(input, target, company_keys, company_dict, macro_keys, macro_dict, **dataloader_kwargs)
        else:
            test_data = prepare_cross_sectional_inference_loader(input, target)
    else:
        input = test_input
        company_keys = test_company_keys
        macro_keys = test_macro_keys
        target = test_target
        mapping_df = test_mapping_df
        cum_label = cum_label_test.values
        if model_configs.use_macro:
            test_data = prepare_cross_sectional_inference_loader(input, target, company_keys, company_dict, macro_keys, macro_dict, **dataloader_kwargs)
        else:
            test_data = prepare_cross_sectional_inference_loader(input, target)

    # Inference
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_data, desc=f"Cross Sectional Inference", leave=False, ncols=80):
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
    calc_data_g_list = []

    for horizon in horizon_list:
        
        mask_invalid = (cum_label[:,horizon] != -1)
        dates_mask = (dates >= pd.to_datetime('1994-01-01'))
        final_mask = mask_invalid&dates_mask
        
        # Dataframe with 3 columns (dates, labels, and predictions)
        calc_data = pd.DataFrame(dates[final_mask], columns=['date'])
        calc_data['{}M_l'.format(horizon + 1)] = cum_label[final_mask, horizon]
        calc_data['{}M_p'.format(horizon + 1)] = cumulative_default_probability[final_mask, horizon]
        if calc_data.shape[0] != 0:
            calc_data_g = calc_data.groupby('date').sum()
            calc_data_g_list.append(calc_data_g)

    return calc_data_g_list


def main(args):

    horizon_list = [0, 2, 5, 11, 23, 35]
    result_list = []
   
    for run in range(args.runs):
        result_list.append(eval(args, run))

    avg_result_list = []
    for dfs in zip(*result_list):
        avg_df = sum(dfs) / len(dfs)
        avg_result_list.append(avg_df)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for i, h in enumerate(horizon_list):
        ax = axes[i]
        result_df = avg_result_list[i]
        ax.plot(result_df.index.to_numpy(), result_df['{}M_p'.format(h + 1)].values, color="red", linewidth=1, label="Predict")
        ax.bar(result_df.index.to_numpy(), result_df['{}M_l'.format(h + 1)].values, width=1, align='center', color="#00A2FF", edgecolor="#00A2FF",alpha=0.7, label="Observed")
        ax.set_title(f"{h + 1} m")
        ax.grid(True)
    
    result_df.to_csv(f"experiments/{args.exp_name}_cross_sectional_cnt.csv")
    plt.tight_layout()
    plt.savefig(f"experiments/{args.exp_name}_count.png")
    print(f"Count plot saved to experiments/{args.exp_name}_count.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment parameters
    parser.add_argument(
        "--exp_name", required=True, type=str, default=None, help="Name of experiment"
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of run of cross-sectional experiment with different random seeds"
    )
    parser.add_argument(
        "--inference_type", type=str, default="test", choices=['test', 'valid'], help="Type of inference to perform"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="The specific number of gpu"
    )
  
    args = parser.parse_args()

    # Print Config
    print(args)

    main(args)