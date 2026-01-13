import sys
import os
import wandb
import argparse
from easydict import EasyDict
from transformers import get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from data.datasethdf5_torch import HDF5TorchDataset
from utils.set_seed import set_seed
from utils.read_config_file import read_config_file
from utils.quantile_losses import AllQuantileLoss, CosineQuantileLoss
from utils.model_setup_utils import init_weights, linear_warmup_step_decay_scheduler
from models.cqr_nn_angles import *


def main():
    parser = argparse.ArgumentParser(description="Run train and/or test.")
    parser.add_argument(
        "--model_cqra_cfg",
        type=str,
        required=True,
        help="Name of model cfg file.",
    )
    parser.add_argument(
        "--train_file_path",
        type=str,
        required=True,
        # default="train_file.hdf5",
        help="Path of training hdf5 file.",
    )
    parser.add_argument(
        "--val_file_path",
        type=str,
        # default="val_file.hdf5",
        help="Path of val hdf5 file.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        # default="/checkpoint_folder/",
        help="Path of chekcpoints directory.",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default="/wandb/",
        help="Path of wandb folder.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.9,
        help="Overall quantile.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay optimizer.",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cos_sched",
        help="Type of scheduler to use. Options: 'cos_sched', 'lambda_lr', 'step_lr'.",
    )
    parser.add_argument(
        "--sched_step_size",
        type=int,
        default=None,
        help="Scheduler step size if needed",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=None,
        help="Scheduler step size if needed",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Init hidden dim, if needed",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        # default="angles_and_length",
        help="choose among: 'angles_and_length', 'angles'",
    )
    parser.add_argument(
        "--flag_allquantiles",
        type=bool,
        default=True,
        help="Flag to use all quantiles",
    )
    # Setup cfg structure
    cfg = vars(parser.parse_args())
    cfg = read_config_file("./configs/cqra_training/", cfg["model_cqra_cfg"], cfg)
    cfg = EasyDict(cfg)
    wandb_run_name = f"{cfg.model_cqra_cfg}_Ep{str(cfg.num_epochs)}_Lr{str(cfg.lr)}_BS{str(cfg.batch_size)}_Sch{str(cfg.scheduler_type)}"
    cfg.quantile_low = (1.0 - cfg.quantile) / 2  # (1-0.9)/2 = 0.05
    cfg.quantile_high = 1.0 - (1.0 - cfg.quantile) / 2  # 1 - (1-0.9)/2 = 0.95

    if cfg.test_type == "angles_and_length":
        wandb_project = f"CorrectDiff_cp_angles_and_length_NoStepFilter_{cfg.quantile_low:.2f}_{cfg.quantile_high:.2f}"
        cfg.checkpoint_dir = os.path.join(
            cfg.checkpoint_dir,
            f"angles_and_length/quant_{cfg.quantile_low:.2f}_{cfg.quantile_high:.2f}",
            cfg.net_type,
            wandb_run_name,
        )
    elif cfg.test_type == "angles":
        wandb_project = f"cp_angles_{cfg.quantile_low:.2f}_{cfg.quantile_high:.2f}"
        cfg.checkpoint_dir = os.path.join(
            cfg.checkpoint_dir,
            f"angles_cosine_difference_loss/quant_{cfg.quantile_low:.2f}_{cfg.quantile_high:.2f}",
            cfg.net_type,
            wandb_run_name,
        )  # /angles_no_step_filter/
    if not os.path.exists(cfg.checkpoint_dir):
        os.makedirs(cfg.checkpoint_dir)

    # Setup wandb logging
    wandb.init(
        project=wandb_project,
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
    set_seed(42)

    train_dataset = HDF5TorchDataset(cfg.train_file_path)
    val_dataset = HDF5TorchDataset(cfg.val_file_path)

    train_data_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2
    )

    # Initialize model, optimizer
    model_cls = getattr(
        sys.modules[__name__], cfg.net_type, None
    )  # Look up model dynamically
    if model_cls is None:
        raise ValueError(f"Model {cfg.net_type} not found in the current script.")
    if cfg.test_type == "angles_and_length":
        num_tests = 2
    elif cfg.test_type == "angles":
        num_tests = 1
    model = model_cls(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_quantiles=2,
        num_tests=num_tests,
    ).to(device)
    init_weights(model)
    model = nn.DataParallel(model)

    if cfg.weight_decay != None:
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    if cfg.scheduler_type == "cos_sched":
        num_warmup_steps = int(cfg.lr_warmup_perc * cfg.num_epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, cfg.num_epochs
        )
    elif cfg.scheduler_type == "step_lr":
        if cfg.lr_gamma == None:
            print("setting up default lr_gamma to 0.1")
            cfg.lr_gamma = 0.1
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.sched_step_size, gamma=cfg.lr_gamma
        )
    elif cfg.scheduler_type == "lambda_lr":
        scheduler = linear_warmup_step_decay_scheduler(
            optimizer,
            cfg.lr_warmup_perc,
            cfg.num_epochs,
            cfg.sched_step_size,
            cfg.lr_gamma,
        )
    if cfg.test_type == "angles_and_length":
        quantile_loss = AllQuantileLoss([cfg.quantile_low, cfg.quantile_high])
    elif cfg.test_type == "angles":
        quantile_loss = CosineQuantileLoss([cfg.quantile_low, cfg.quantile_high])
    best_val_loss = float("inf")  # Initialize before the loop

    # Training loop
    for epoch in range(cfg.num_epochs):
        model.train()
        train_epoch_loss = 0.0
        for (
            batch_data,
            _,
            _,
        ) in train_data_loader:
            features = batch_data["features"].to(device)
            gt_angle = batch_data["gt_angle"].to(device)
            pred_angle = batch_data["pred_angle"].to(device)

            if cfg.test_type == "angles_and_length":
                gt_deltax = batch_data["gt_deltax"].to(device)
                gt_deltay = batch_data["gt_deltay"].to(device)
                # Compute lengths
                gt_length = torch.sqrt(gt_deltax**2 + gt_deltay**2)
                target_length = (gt_length).unsqueeze(-1)
                target_angle = torch.atan2(
                        torch.sin(gt_angle - pred_angle),
                        torch.cos(gt_angle - pred_angle),
                    ).unsqueeze(-1)
                target = torch.cat((target_angle, target_length), dim=-1)

            # Normalize the angle difference to [-π, π]
            elif cfg.test_type == "angles":
                target = torch.atan2(
                    torch.sin(gt_angle - pred_angle),
                    torch.cos(gt_angle - pred_angle),
                ).unsqueeze(-1)
            optimizer.zero_grad()

            # Forward pass
            predictions = model(features)

            # Compute quantile losses
            loss = quantile_loss(predictions, target.unsqueeze(2))

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_epoch_loss += loss.item()
            # Get current learning rate from the optimizer
            current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()
        # Validation loop
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for batch_data_val, _, _ in val_data_loader:
                features_val = batch_data_val["features"].to(device)
                gt_angle_val = batch_data_val["gt_angle"].to(device)
                pred_angle_val = batch_data_val["pred_angle"].to(device)
                # Forward pass
                predictions_val = model(features_val)

                if cfg.test_type == "angles_and_length":
                    gt_deltax_val = batch_data_val["gt_deltax"].to(device)
                    gt_deltay_val = batch_data_val["gt_deltay"].to(device)
                    # Compute lengths
                    gt_length_val = torch.sqrt(gt_deltax_val**2 + gt_deltay_val**2)
                    target_length = (gt_length_val).unsqueeze(-1)
                    target_angle_val = torch.atan2(
                            torch.sin(gt_angle_val - pred_angle_val),
                            torch.cos(gt_angle_val - pred_angle_val),
                        ).unsqueeze(-1)
                    target_val = torch.cat((target_angle_val, target_length), dim=-1)
                # Normalize the angle difference to [-π, π]
                elif cfg.test_type == "angles":
                    target_val = torch.atan2(
                        torch.sin(gt_angle_val - pred_angle_val),
                        torch.cos(gt_angle_val - pred_angle_val),
                    ).unsqueeze(-1)
                # Compute quantile losses
                loss = quantile_loss(predictions_val, target_val.unsqueeze(2))
                val_epoch_loss += loss.item()

        val_epoch_loss /= len(val_data_loader)
        print(
            f"Epoch {epoch+1}/{cfg.num_epochs}, Validation Loss: {val_epoch_loss}, Train Loss: {train_epoch_loss / len(train_data_loader)}"
        )

        # Save the best model
        if epoch == 0 or val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(
                model.state_dict(),
                os.path.join(
                    cfg.checkpoint_dir,
                    f"best_model_{str(cfg.quantile_low)}_{cfg.quantile_high}_epoch{str(epoch)}.pth",
                ),
            )
            print(f"Best model saved with validation loss: {best_val_loss}")
        wandb.log(
            {
                "learning_rate_cqr": current_lr,
                "train_loss_cqr": train_epoch_loss / len(train_data_loader),
                "val_loss_cqr": val_epoch_loss,
            }
        )
    torch.save(
        model.state_dict(),
        os.path.join(
            cfg.checkpoint_dir,
            f"epoch_{epoch}_model_{str(cfg.quantile_low)}_{cfg.quantile_high}.pth",
        ),
    )


if __name__ == "__main__":
    main()
