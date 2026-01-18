import os
import sys
import torch
import tqdm
import argparse
import json
import pickle
from utils.metric import AUCMetric
from utils.logger import Logger
from utils.setup_model import setup_model
from utils.configs import get_dataloader_kwargs
from utils.generator import experiment_name
from dataloader import load_raw_data, prepare_cross_sectional_labeled_loader
from layers.loss import CustomCrossEntropy


def train(args, run):
    if args.seed:
        torch.manual_seed(args.seed)

    # Set up experiment folder
    exp_name = experiment_name(args)
    exp_name = f"{exp_name}_{run}"
    try:
        os.makedirs(f"runs/{exp_name}")
    except FileExistsError:
        sys.exit(f"Run directory 'runs/{exp_name}' already exists. Aborting.")
    with open(f"runs/{exp_name}/args.pkl", "wb") as f:
        pickle.dump(args, f)
    print(f"Training: {exp_name}")

    # Set up device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = setup_model(args)
    model.to(device)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Set up loss function
    criterion = CustomCrossEntropy()

    # Set up logger
    logger = Logger(exp_name)

    # Load data
    train_input, train_company_keys, train_macro_keys, train_target, train_mapping_df, cum_label_train, \
    test_input, test_company_keys, test_macro_keys, test_target, test_mapping_df, cum_label_test, \
    company_dict, macro_dict = load_raw_data(args.data_dir, args.level)
    
    dataloader_kwargs = get_dataloader_kwargs(args)
    if args.use_macro:
        train_data, val_data = prepare_cross_sectional_labeled_loader(
            train_input, train_target, train_mapping_df, run, train_company_keys, company_dict, train_macro_keys, macro_dict, **dataloader_kwargs
        )
    else:
        train_data, val_data = prepare_cross_sectional_labeled_loader(
            train_input, train_target, train_mapping_df, run
        )

    # Train initialization
    train_auc = AUCMetric(["all", 1, 6, 12, 48])
    val_auc = AUCMetric(["all", 1, 6, 12, 48])
    patience = args.patience
    max_val_auc = 0

    # Train main
    for i in range(args.epoch):
        train_auc.reset_state()
        val_auc.reset_state()
        train_loss = 0
        model.train()
        for batch in tqdm.tqdm(train_data, desc=f"Train Epoch {i}", leave=False, ncols=80):
            optimizer.zero_grad()
            *inputs, y = batch
            inputs = [input.to(device) for input in inputs]
            y = y.to(device)
            if args.use_macro:
                x, c, m, m_mask = inputs
                y_pred = model(x, c, m, m_mask)
            else:
                x = inputs[0]
                y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            train_auc.update_state(y_pred, y)
        
        train_loss /= len(train_data)
        train_auc_value = train_auc.result()
        logger.log(epoch=i, action="Train", loss=train_loss, aucs=train_auc_value, show=args.log)
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in tqdm.tqdm(val_data, desc=f"Val Epoch {i}", leave=False, ncols=80):
                *inputs, y = batch
                inputs = [input.to(device) for input in inputs]
                y = y.to(device)
                if args.use_macro:
                    x, c, m, m_mask = inputs
                    y_pred = model(x, c, m, m_mask)
                else:
                    x = inputs[0]
                    y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
                val_auc.update_state(y_pred, y)
            val_loss /= len(val_data)
            val_auc_value = val_auc.result()

            better = val_auc_value["all"] > max_val_auc
            logger.log(epoch=i, action="Valid", loss=val_loss, aucs=val_auc_value, better=better, show=args.log)

            if better:
                max_val_auc = val_auc_value["all"]
                patience = args.patience
                torch.save(
                    {
                        "epoch": i,
                        "model_type": args.model_type,
                        "feature_size": args.feature_size,
                        "window_size": args.window_size,
                        "horizon_size": args.horizon_size,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_auc": val_auc_value,
                        "seed": torch.initial_seed(),
                    },
                    f"runs/{exp_name}/best_ckpt.pt",
                )
                json.dump(
                    {
                        "exp_name": args.exp_name,
                        "model_type": args.model_type,
                        "feature_size": args.feature_size,
                        "window_size": args.window_size,
                        "horizon_size": args.horizon_size,
                        "best_epoch": i,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_auc": val_auc_value,
                        "seed": torch.initial_seed(),
                    },
                    open(f"runs/{exp_name}/result.json", "w"),
                    indent=4,
                )
            else:
                patience -= 1
                if patience == 0:
                    print(f"Early stop at epoch {i}. Best epoch: {i - args.patience}, Best AUC: {max_val_auc}")
                    logger.log(
                        epoch=i, action="Valid", loss=val_loss, aucs=val_auc_value, better=better, show=args.log
                    )
                    print(f"Best epoch: {i - args.patience}", f"Best AUC: {max_val_auc}")
                    break

def main(args):
    for run in range(args.runs):
        train(args, run)

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
        "--window_size", type=int, default=12, help="The sequence length of company features"
    )
    parser.add_argument(
        "--feature_size", type=int, default=16, help="The size of company features"
    )
    parser.add_argument(
        "--horizon_size", type=int, default=60, help="The length of prediction horizon"
    )
    parser.add_argument(
        "--log", action="store_true", help="Whether to log the training process"
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of run of cross-sectional experiment with different random seeds"
    )
    parser.add_argument(
        "--use_macro", action="store_true", help="Whether to use macro features"
    )
    parser.add_argument(
        "--n_macro", type=int, default=1, help="Number of macro to use"
    )
    parser.add_argument(
        "--level", type=int, default=0, help="ablation study tag"
    )
    parser.add_argument(
        "--backbone", type=str, default="mlp", choices=["fim", "mlp", "transformer"], help="backbone model for dfn and dfna3"
    )
    
    # Model parameters
    parser.add_argument(
        "--model_type", type=str, default="mlp", choices=["fim", "mlp", "transformer", "dfn", "dfna3"], help="Type of model to use"
    )
    parser.add_argument(
        "--CRI", dest="CRI", action="store_true", help="Enable CRI"
    )
    parser.add_argument(
        "--no_CRI", dest="CRI", action="store_false", help="Disable CRI (default: enabled)"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="The random seed to use"
    )
    parser.add_argument(
        "--dropout", type=float, default=0, help="Dropout rate"
    )

    # Training parameters
    parser.add_argument(
        "--gpu_id", type=int, default=1, help="The speicific number of gpu"
    )
    parser.add_argument(
        "--epoch", type=int, default=1000, help="Epoch"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2048, help="The size of batch for training"
    )

    # Eraly stop parameters
    parser.add_argument(
        "--early_stop", dest="early_stop", action="store_true", help="Enable early stopping (default: enabled)"
    )
    parser.add_argument(
        "--no_early_stop", dest="early_stop", action="store_false", help="Disable early stopping"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="How many times we tolerate the model for no improvement"
    )

    parser.set_defaults(
        CRI=True,
        early_stop=True
    )

    args = parser.parse_args()
    args.use_macro = True if args.model_type in ["dfn", "dfna3"] else False
    if args.level == 3 and args.model_type != "dfna3":
        raise ValueError(
            f"Invalid config: level=3 requires model_type='dfna3' (got '{args.model_type}')."
        )

    # Print config
    print(args)

    main(args)