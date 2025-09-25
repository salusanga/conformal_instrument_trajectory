# %% import torch
import sys
import os
import wandb
import numpy as np
import argparse
from easydict import EasyDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from data.datasethdf5_torch import HDF5TorchDataset
from data.datasethdf5_helper_functions import split_dataset
from utils.set_seed import set_seed
from utils.read_config_file import read_config_file
from utils.conformal_prediction_step import (
    compute_abs_angles_errors,
    compute_abs_length_errors,
    get_quantile,
    cp_compute_corrected_quantiles_and_coverage,
    cqr_compute_corrected_quantiles_and_coverage,
)
from utils.print_statistics import *
from utils.quantile_losses import AllQuantileLoss, CosineQuantileLoss

from models.cqr_nn_angles import *
import pandas as pd


def find_best_checkpoint(checkpoint_dir, quantile_low, quantile_high):
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("best_model_")
    ]
    if checkpoint_files:
        highest_epoch = max(
            int(f.split("_epoch")[1].split(".pth")[0]) for f in checkpoint_files
        )
        checkpoint_name = os.path.join(
            checkpoint_dir,
            f"best_model_{str(quantile_low)}_{quantile_high}_epoch{str(highest_epoch)}.pth",
        )
    else:
        checkpoint_name = os.path.join(
            checkpoint_dir,
            f"best_model_{str(quantile_low)}_{quantile_high}_epoch0.pth",
        )
    return checkpoint_name


def find_last_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_")]
    if checkpoint_files:
        latest_checkpoint = max(
            checkpoint_files, key=lambda x: int(x.split("_")[1].split(".pth")[0])
        )
        checkpoint_name = os.path.join(checkpoint_dir, latest_checkpoint)
    else:
        checkpoint_name = os.path.join(checkpoint_dir, "epoch_0.pth")
    return checkpoint_name


def loop_quantiles_inference(data_loader, model_cqr, quantile_loss, split, cfg):
    (
        all_predictions,
        all_targets,
        all_patients,
        all_timestamps,
        all_gt_vector_angles,
        all_pred_vector_angles,
    ) = ([], [], [], [], [], [])
    tot_loss = 0
    for (
        batch_data,
        patients,
        timestamps,
    ) in (
        data_loader
    ):  # Assuming dataloader yields (features, gt_angle, gt_deltax, gt_deltay)
        features = batch_data["features"].to(next(model_cqr.parameters()).device)
        gt_angle = batch_data["gt_angle"].to(next(model_cqr.parameters()).device)
        pred_angle = batch_data["pred_angle"].to(next(model_cqr.parameters()).device)
        gt_deltax = batch_data["gt_deltax"].to(next(model_cqr.parameters()).device)
        gt_deltay = batch_data["gt_deltay"].to(next(model_cqr.parameters()).device)

        # Compute lengths
        gt_length = torch.sqrt(gt_deltax**2 + gt_deltay**2)
        target_length = gt_length.unsqueeze(-1)

        # Compute angle differences
        target_angle = torch.atan2(
                torch.sin(gt_angle - pred_angle),
                torch.cos(gt_angle - pred_angle),
            ).unsqueeze(-1)

        # Concatenate angle differences and lengths
        target = torch.cat((target_angle, target_length), dim=-1)

        # Forward pass
        predictions_cqr = model_cqr(features)

        # Store predictions and targets
        all_predictions.append(predictions_cqr.cpu().detach().numpy())
        all_targets.append(target.cpu().detach().numpy())
        all_patients.extend(patients)
        all_timestamps.extend(timestamps)
        all_gt_vector_angles.extend(gt_angle.unsqueeze(-1).cpu().detach().numpy())
        all_pred_vector_angles.extend(pred_angle.unsqueeze(-1).cpu().detach().numpy())

        # Compute quantile losses
        loss = quantile_loss(predictions_cqr, target.unsqueeze(2))

        # Accumulate loss
        tot_loss += loss.item()

    print(f"{split} quantile loss: {tot_loss / len(data_loader):.3f}")

    # Convert lists to numpy arrays
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_gt_vector_angles = np.concatenate(all_gt_vector_angles, axis=0)
    all_pred_vector_angles = np.concatenate(all_pred_vector_angles, axis=0)

    return (
        all_predictions,
        all_targets,
        all_patients,
        all_timestamps,
        all_gt_vector_angles,
        all_pred_vector_angles,
    )


def append_to_excel(file_name, list_to_append):
    columns = [
        "Quantile",
        "Angle Quantile CP",
        "Lower Bound CQR Angles",
        "Upper Bound CQR Angles",
        "Length Quantile CP",
        "Lower Bound CQR Lengths",
        "Upper Bound CQR Lengths",
    ]
    list_to_append = [
        x.item() if isinstance(x, torch.Tensor) else x for x in list_to_append
    ]
    df = pd.DataFrame([list_to_append], columns=columns)
    if not os.path.isfile(file_name):
        df.to_excel(file_name, index=False)
    else:
        existing_df = pd.read_excel(file_name)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_excel(file_name, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run train and/or test.")
    parser.add_argument(
        "--model_cqra_cfg",
        type=str,
        # default="cfg_1_angle",
        help="Name of model cfg file.",
    )
    parser.add_argument(
        "--test_file_path",
        type=str,
        required=True,
        # default="test_file.hdf5",
        help="Path of test file.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        # default="/checkpoint_folder/",
        required=True,
        help="Path of chekcpoints directory.",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default="/wandb/",
        help="Path of wandb folder.",
    )
    parser.add_argument(
        "--which_checkpoint",
        type=str,
        default="best",
        help="Which checkpoint to parse",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        required=True,
        default=0.9,
        help="Low quantile.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay optimizer.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=7,
        help="seeds.",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        # default="angles_and_length",
        required=True,
        help="choose among: 'angles_and_length', 'angles'",
    )
    parser.add_argument(
        "--split_found",
        type=bool,
        default=False,
        help="Flag to check if split with correct test patient is found.",
    )
    parser.add_argument(
        "--patient_to_select",
        type=int,
        default=129,  # 133,
        help="patient to select for plotting.",
    )
    parser.add_argument(
        "--timestamp_to_select",
        type=int,
        default=257,  # 48337,
        help="patient to select for plotting.",
    )
    # Setup cfg structure
    cfg = vars(parser.parse_args())
    cfg = read_config_file("./configs/cqra_training/", cfg["model_cqra_cfg"], cfg)
    cfg = EasyDict(cfg)
    cfg.quantile_low = (1.0 - cfg.quantile) / 2  # (1-0.9)/2 = 0.05
    cfg.quantile_high = 1.0 - (1.0 - cfg.quantile) / 2  # 1 - (1-0.9)/2 = 0.95
    print(
        f" ########### RUNNING CONFORMAL PREDICTION FOR TARGET COVERAGE = {cfg.quantile}. ###########"
    )

    if "angles_no_step_filter" in cfg.checkpoint_dir:
        wandb_project_name = f"evaluation_{cfg.quantile_low:.2f}_{cfg.quantile_high:.2f}_{str(int(cfg.num_seeds))}Seeds_angles_quantile_loss"
    elif "angles_cosine_loss" in cfg.checkpoint_dir:
        wandb_project_name = f"evaluation_{cfg.quantile_low:.2f}_{cfg.quantile_high:.2f}_{str(int(cfg.num_seeds))}Seeds_angles_cosine_loss"
    elif "angles_and_length" in cfg.checkpoint_dir:
        wandb_project_name = f"evaluation_CorrectDiff_{cfg.quantile_low:.2f}_{cfg.quantile_high:.2f}_{str(int(cfg.num_seeds))}Seeds_angles_and_length"

    wandb_run_name = f"{cfg.model_cqra_cfg}_Ep{str(cfg.num_epochs)}_Lr{str(cfg.lr)}_BS{str(cfg.batch_size)}_Sch{str(cfg.scheduler_type)}"

    # # Setup wandb logging
    wandb.init(
        project=wandb_project_name,
        name=wandb_run_name,
        dir=cfg.wandb_dir,
        config=cfg,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Find the pre-saved checkpoint for the quantile regression with the highest epoch number
    cfg.checkpoint_dir = os.path.join(
        cfg.checkpoint_dir,
        f"quant_{cfg.quantile_low:.2f}_{cfg.quantile_high:.2f}",
        cfg.net_type,
        wandb_run_name,
    )
    if cfg.which_checkpoint == "best":
        cfg.checkpoint_name = find_best_checkpoint(
            cfg.checkpoint_dir, cfg.quantile_low, cfg.quantile_high
        )
    if cfg.which_checkpoint == "last":
        cfg.checkpoint_name = find_last_checkpoint(cfg.checkpoint_dir)

    print(f"Using checkpoint: {cfg.checkpoint_name}.")

    # Initialize model, optimizer
    model_cls = getattr(
        sys.modules[__name__], cfg.net_type, None
    )  # Look up model dynamically
    if model_cls is None:
        raise ValueError(f"Model {cfg.net_type} not found in the current script.")
    model_cqr = model_cls(
        input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim, num_quantiles=2, num_tests=2
    ).to(device)
    model_cqr = nn.DataParallel(model_cqr)
    model_cqr.load_state_dict(
        torch.load(cfg.checkpoint_name, map_location=device, weights_only=True)
    )
    model_cqr.eval()
    cp_coverage_angle_list = []
    cp_coverage_length_list = []
    cp_int_width_angle_list = []
    cp_int_width_length_list = []

    cqr_coverage_angle_list = []
    cqr_coverage_length_list = []
    cqr_int_width_angle_list = []
    cqr_int_width_length_list = []

    cp_NON_corrected_overall_coverage_list = []
    cp_NON_corrected_overall_width_list = []

    cp_CORRECTED_overall_list = []

    cqr_NON_corrected_overall_coverage_list = []
    cqr_NON_corrected_overall_width_list = []

    cqr_CORRECTED_overall_list = []

    test_abs_angles_errors_list = []
    test_abs_length_errors_list = []
    test_length_gt_avg_list = []

    # Loss setup
    if (
        "angles_no_step_filter" in cfg.checkpoint_dir
        or "angles_and_length" in cfg.checkpoint_dir
    ):
        quantile_loss = AllQuantileLoss([cfg.quantile_low, cfg.quantile_high])
    elif "angles_cosine_loss" in cfg.checkpoint_dir:
        quantile_loss = CosineQuantileLoss([cfg.quantile_low, cfg.quantile_high])
    print("Number of seeds:", cfg.num_seeds)

    for seed in range(int(cfg.num_seeds)):
        set_seed(seed)
        print("Seed: ", seed)
        test_dataset = HDF5TorchDataset(
            cfg.test_file_path,
        )
        val_data_loader, test_data_loader, _, subset_test = split_dataset(
            test_dataset, split_ratio_val=0.6
        )
        all_predictions_val, all_targets_val, _, _, _, _ = loop_quantiles_inference(
            val_data_loader, model_cqr, quantile_loss, "Val", cfg
        )
        (
            all_predictions_test,
            all_targets_test,
            all_patients_test,
            all_timestamps_test,
            all_gt_vector_angles_test,
            all_pred_vector_angles_test,
        ) = loop_quantiles_inference(
            test_data_loader, model_cqr, quantile_loss, "Test", cfg
        )

        ### CP ###
        print("Test statistics:")
        test_abs_angles_errors = compute_abs_angles_errors(test_data_loader)
        test_abs_length_errors, test_length_gt_avg = compute_abs_length_errors(
            test_data_loader
        )
        test_abs_angles_errors_list.append(test_abs_angles_errors.mean().item())
        test_abs_length_errors_list.append(test_abs_length_errors.mean().item())
        test_length_gt_avg_list.append(test_length_gt_avg)
        print("Val statistics:")
        val_abs_angles_errors = compute_abs_angles_errors(val_data_loader)
        val_abs_length_errors, _ = compute_abs_length_errors(val_data_loader)
        angle_quantile_cp = get_quantile(
            val_abs_angles_errors,
            torch.tensor(1.0 - cfg.quantile),
            val_abs_angles_errors.shape[0],
        )
        length_quantile_cp = get_quantile(
            val_abs_length_errors,
            torch.tensor(1.0 - cfg.quantile),
            val_abs_length_errors.shape[0],
        )

        # CP coverage and interval size, separately for angle and length
        coverage_angle_vector_cp = (
            test_abs_angles_errors < angle_quantile_cp.item()
        ).float()
        coverage_angle = coverage_angle_vector_cp.mean()
        coverage_length_vector_cp = (
            test_abs_length_errors < length_quantile_cp.item()
        ).float()
        coverage_length = coverage_length_vector_cp.mean()

        cp_coverage_angle_list.append(coverage_angle)
        cp_coverage_length_list.append(coverage_length)
        cp_int_width_angle_list.append(angle_quantile_cp.item() * 2)
        cp_int_width_length_list.append(length_quantile_cp.item() * 2)

        # CP coverage and interval size, NON-corrected for angle and length together
        cp_NON_corrected_overall_coverage = (
            (
                (test_abs_angles_errors < angle_quantile_cp.item())
                & (test_abs_length_errors < length_quantile_cp.item())
            )
            .float()
            .mean()
        )
        cp_NON_corrected_overall_coverage_list.append(cp_NON_corrected_overall_coverage)
        cp_NON_corrected_overall_width_list.append(
            angle_quantile_cp.item() + length_quantile_cp.item()
        )

        # CP coverage and interval size, CORRECTED for angle and length together
        corrected_cp_dict = cp_compute_corrected_quantiles_and_coverage(
            test_abs_angles_errors,
            test_abs_length_errors,
            val_abs_angles_errors,
            val_abs_length_errors,
            cfg.quantile,
        )

        cp_CORRECTED_overall_list.append(corrected_cp_dict)

        ### CQR ###
        # Compute conformity scores for angles, separately
        val_conformity_scores_cqr_angles = torch.max(
            torch.tensor(all_predictions_val)[:, 0, 0]
            - torch.tensor(all_targets_val)[:, 0],
            torch.tensor(all_targets_val)[:, 0]
            - torch.tensor(all_predictions_val)[:, 0, 1],
        )
        angle_quantile_cqr = get_quantile(
            val_conformity_scores_cqr_angles,
            torch.tensor(1.0 - cfg.quantile),
            val_conformity_scores_cqr_angles.shape[0],
        )
        lower_bound_cqr_angles = (
            all_predictions_test[:, 0, 0] - angle_quantile_cqr.item()
        )
        upper_bound_cqr_angles = (
            all_predictions_test[:, 0, 1] + angle_quantile_cqr.item()
        )
        coverage_angle_vector_cqr = (
            (all_gt_vector_angles_test - all_pred_vector_angles_test + np.pi)
            % (2 * np.pi)
            - np.pi
            > lower_bound_cqr_angles
        ) & (
            (all_gt_vector_angles_test - all_pred_vector_angles_test + np.pi)
            % (2 * np.pi)
            - np.pi
            < upper_bound_cqr_angles
        )

        coverage_angle_cqr = np.mean(coverage_angle_vector_cqr)
        intervals_width_cqr_angles = upper_bound_cqr_angles - lower_bound_cqr_angles

        mean_interval_width_angle_cqr = np.mean(intervals_width_cqr_angles)
        cqr_coverage_angle_list.append(coverage_angle_cqr)
        cqr_int_width_angle_list.append(mean_interval_width_angle_cqr)

        # Compute conformity scores for length, separately
        val_conformity_scores_cqr_lengths = torch.max(
            torch.tensor(all_predictions_val)[:, 1, 0]
            - torch.tensor(all_targets_val)[:, 1],
            torch.tensor(all_targets_val)[:, 1]
            - torch.tensor(all_predictions_val)[:, 1, 1],
        )
        length_quantile_cqr = get_quantile(
            val_conformity_scores_cqr_lengths,
            torch.tensor(1.0 - cfg.quantile),
            val_conformity_scores_cqr_lengths.shape[0],
        )
        lower_bound_cqr_lengths = (
            all_predictions_test[:, 1, 0] - length_quantile_cqr.item()
        )
        upper_bound_cqr_lengths = (
            all_predictions_test[:, 1, 1] + length_quantile_cqr.item()
        )
        coverage_length_vector_cqr = (
            all_targets_test[:, 1] > lower_bound_cqr_lengths
        ) & (all_targets_test[:, 1] < upper_bound_cqr_lengths)
        coverage_length_cqr = np.mean(coverage_length_vector_cqr)
        intervals_width_cqr_lengths = upper_bound_cqr_lengths - lower_bound_cqr_lengths
        mean_interval_width_length_cqr = np.mean(intervals_width_cqr_lengths)
        cqr_coverage_length_list.append(coverage_length_cqr)
        cqr_int_width_length_list.append(mean_interval_width_length_cqr)


        # CQR coverage and interval size, NON-corrected for angle and length together
        cqr_NON_corrected_overall_coverage = np.mean(
            (all_targets_test[:, 0] > lower_bound_cqr_angles)
            & (all_targets_test[:, 0] < upper_bound_cqr_angles)
            & (all_targets_test[:, 1] > lower_bound_cqr_lengths)
            & (all_targets_test[:, 1] < upper_bound_cqr_lengths)
        )
        cqr_NON_corrected_overall_coverage_list.append(
            cqr_NON_corrected_overall_coverage
        )
        cqr_NON_corrected_overall_width_list.append(
            (mean_interval_width_angle_cqr + mean_interval_width_length_cqr) / 2
        )

        # CQR coverage and interval size, CORRECTED for angle and length together
       dict_correct_cqr = cqr_compute_corrected_quantiles_and_coverage(
            all_targets_test,
            all_predictions_test,
            val_conformity_scores_cqr_angles,
            val_conformity_scores_cqr_lengths,
            all_gt_vector_angles_test,
            all_pred_vector_angles_test,
            cfg.quantile,
        )

        cqr_CORRECTED_overall_list.append(dict_correct_cqr)



    print_statistics(
        cp_coverage_angle_list,
        cp_coverage_length_list,
        cp_int_width_angle_list,
        cp_int_width_length_list,
        cqr_coverage_angle_list,
        cqr_coverage_length_list,
        cqr_int_width_angle_list,
        cqr_int_width_length_list,
        cp_NON_corrected_overall_coverage_list,
        cp_NON_corrected_overall_width_list,
        cp_CORRECTED_overall_list,
        cqr_NON_corrected_overall_coverage_list,
        cqr_NON_corrected_overall_width_list,
        cqr_CORRECTED_overall_list,
        test_abs_angles_errors_list,
        test_abs_length_errors_list,
        test_length_gt_avg_list,
    )


if __name__ == "__main__":
    main()
