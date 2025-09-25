# %%
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data.datasethdf5_torch import HDF5TorchDataset
from utils.set_seed import set_seed
import numpy as np
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histograms_angles_length(
    scores: torch.Tensor, filename: str = "histograms.png"
):
    """
    Generates histogram probability plots for each column of a torch tensor and saves them in one PNG file.

    Args:
        scores (torch.Tensor): Tensor of shape [num_samples, 2]
        filename (str): Path to save the output image
    """
    assert scores.shape[1] == 2, "Input tensor must have shape [num_samples, 2]"

    scores = scores.numpy()  # Convert to NumPy for plotting
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

    for i in range(2):
        sns.histplot(scores[:, i], kde=True, bins=30, ax=axes[i], stat="probability")
        axes[i].set_title(f"Histogram of Column {i+1}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Probability")

    plt.savefig(filename, dpi=300)
    plt.close()


def compute_abs_errors_sin_cos(dataloader):
    """
    Computes absolute errors per sample for sin and cos predictions.

    Args:
    - dataloader (DataLoader): Torch dataloader object.

    Returns:
    - torch.Tensor: Tensor of shape (num_patients, 2) with absolute errors.
                    First column → sin errors, Second column → cos errors.
    """
    abs_errors = []

    for batch_data, patient, timestamp in dataloader:
        # Compute ground truth and predicted sin & cos values
        gt_angle = batch_data["gt_angle"]
        pred_angle = batch_data["pred_angle"]

        gt_sin, gt_cos = torch.sin(gt_angle), torch.cos(gt_angle)
        pred_sin, pred_cos = torch.sin(pred_angle), torch.cos(pred_angle)

        # Compute absolute errors
        abs_sin_error = torch.abs(gt_sin - pred_sin)
        abs_cos_error = torch.abs(gt_cos - pred_cos)

        # Stack sin and cos errors as (batch_size, 2)
        batch_errors = torch.stack([abs_sin_error, abs_cos_error], dim=1)
        abs_errors.append(batch_errors)

    # Concatenate all batches into a single tensor of shape (num_patients, 2)
    return torch.cat(abs_errors, dim=0)


def compute_abs_angles_errors(dataloader):
    """
    Calculates the absolute angle errors between the ground truth (GT) vector and the predicted vector
    for all patients and timestamps in a torch dataloader.

    Args:
    - dataloader (DataLoader): Torch dataloader object containing preprocessed data.

    Returns:
    - torch.Tensor: Tensor of shape (num_patients, 1) containing absolute angle differences (in radians).
    """
    angles = []

    # Iterate over the dataloader
    for batch_data, patient, timestamp in dataloader:
        # Extract angles (shape: [batch_size])
        gt_angles = batch_data["gt_angle"].squeeze()  # In radians
        pred_angles = batch_data["pred_angle"].squeeze()  # In radians

        # Compute absolute angle errors
        angle_differences = torch.abs(
            (gt_angles - pred_angles + np.pi) % (2 * np.pi) - np.pi
        )
        angles.append(angle_differences)
    print("Mean of angles errors", torch.cat(angles).mean())
    # Convert to a single torch tensor of shape (num_patients, 1)
    return torch.cat(angles).unsqueeze(1)


def compute_abs_length_errors(dataloader):
    """
    Calculates the absolute length errors between the ground truth (GT) vector and the predicted vector
    for all patients and timestamps in a torch dataloader.

    Args:
    - dataloader (DataLoader): Torch dataloader object containing preprocessed data.

    Returns:
    - torch.Tensor: Tensor of shape (num_patients, 1) containing absolute length differences.
    """
    lengths = []

    # Iterate over the dataloader
    for batch_data, patient, timestamp in dataloader:
        # Extract deltas (shape: [batch_size])
        gt_deltax = batch_data["gt_deltax"].squeeze()
        gt_deltay = batch_data["gt_deltay"].squeeze()
        pred_deltax = batch_data["pred_deltax"].squeeze()
        pred_deltay = batch_data["pred_deltay"].squeeze()

        # Compute lengths
        gt_length = torch.sqrt(gt_deltax**2 + gt_deltay**2)
        pred_length = torch.sqrt(pred_deltax**2 + pred_deltay**2)

        # Compute absolute length errors
        length_difference = torch.abs(gt_length - pred_length)

        # Append to list
        lengths.append(length_difference)
    print("Mean of length errors", torch.cat(lengths).mean())
    print("Mean of length GT", gt_length.mean())
    # Convert to a single torch tensor of shape (num_patients, 1)
    return torch.cat(lengths).unsqueeze(1), gt_length.mean().item()


def get_quantile(scores: torch.Tensor, alpha: torch.Tensor, n, verbose: bool = False):
    """
    Get conformal quantile from calibration samples.
    Conformal quantile formula: ceil[(1-alpha)(n+1)]/n

    Args:
        scores (Tensor): Conformity scores computed for calibration samples.
        alpha (Tensor): Desired nominal coverage levels (per coordinate).
        n (Tensor): Nr. of calibration samples

    Returns:
        Tensor: Alpha-level quantile of conformity scores.
    """
    q = (torch.ceil((1 - alpha) * (n + 1)) / n).clamp(0, 1)
    if verbose:
        print(f"Nominal quantiles:{1 - alpha}")
    # print(f"Sample-corrected quantiles: {q}")
    return torch.quantile(scores, q.to(scores.dtype), dim=0, interpolation="higher")


def compute_quantile_correct(
    scores: torch.Tensor,
    box_correction: str,
    multiple_tests_num: int,
    nr_samp=None,
    alpha=0.1,
):
    """
    Compute quantile for the desired nominal coverage level with
    different box correction methods (multiple testing corrections).

    Options: naive_max, naive_mean, naive_quant, bonferroni, bonferroni_sidak, holm, holm_sidak, simes, rank_global, rank_coord, score_global
    Of relevance for the paper: bonferroni, rank_global, rank_coord, score_global

    Args:
        scores (Tensor): Conformity scores computed for calibration samples.
        box_correction (str): Type of box correction to apply.
        multiple_tests_num (int): Number of multiple tests. For sin, cos = 2.
        nr_samp (int, optional): Number of calibration samples. Defaults to None.
        alpha (float, optional): Desired nominal coverage level. Defaults to 0.1.

    Returns:
        Tensor: Alpha-level multiple testing-corrected quantile of conformity scores.
    """
    assert 0 <= alpha <= 1, f"Nominal coverage {alpha=} not in [0,1]"
    alpha = torch.tensor([alpha])
    n = torch.tensor(nr_samp if nr_samp is not None else scores.shape[0])
    nr_scores = scores.shape[1]
    quant = torch.empty(size=(nr_scores,), dtype=torch.float32)
    max_angle = torch.max(scores[:, 0])
    max_length = torch.max(scores[:, 1])
    # print(max_angle, max_length)
    scores[:, 0] = scores[:, 0] / torch.pi  # max_angle  # torch.pi
    scores[:, 1] = scores[:, 1] / (torch.sqrt(torch.tensor(2.0)))  # max_length  #

    if box_correction == "naive_max":
        # max over coordinate quantiles as global quantile
        q = get_quantile(scores, alpha, n)
        for s in range(nr_scores // multiple_tests_num):
            i, j = s * multiple_tests_num, s * multiple_tests_num + multiple_tests_num
            quant[i:j] = torch.max(q[:, i:j])

    elif box_correction == "naive_mean":
        # mean over coordinate quantiles as global quantile
        q = get_quantile(scores, alpha, n)
        for s in range(nr_scores // multiple_tests_num):
            i, j = s * multiple_tests_num, s * multiple_tests_num + multiple_tests_num
            quant[i:j] = torch.mean(q[:, i:j])

    elif box_correction == "naive_quant":
        # global quantiles across all coordinate scores concatenated
        for s in range(nr_scores // multiple_tests_num):
            i, j = s * multiple_tests_num, s * multiple_tests_num + multiple_tests_num
            quant[i:j] = get_quantile(scores[:, i:j].flatten(), alpha, n)

    elif box_correction == "bonferroni":
        # FWER Bonferroni correction
        alpha_bonf = alpha / multiple_tests_num
        quant[:] = get_quantile(scores, alpha_bonf, n)

    elif box_correction == "bonferroni_sidak":
        # FWER Bonferroni correction with Sidak improvement
        alpha_bsidak = 1 - (1 - alpha) ** (1 / multiple_tests_num)
        quant[:] = get_quantile(scores, alpha_bsidak, n)
        # print("Corrected Sidak:", quant)

    elif box_correction == "holm":
        # FWER Holm correction
        alpha_holm = torch.zeros(multiple_tests_num)
        for k in range(1, multiple_tests_num + 1):
            alpha_holm[k - 1] = alpha / (multiple_tests_num + 1 - k)
        q = get_quantile(scores, alpha_holm, n)
        for s in range(nr_scores // multiple_tests_num):
            i, j = s * multiple_tests_num, s * multiple_tests_num + multiple_tests_num
            quant[i:j] = torch.diagonal(q[:, i:j])

    elif box_correction == "holm_sidak":
        # FWER Holm correction with Sidak improvement
        alpha_hsidak = torch.zeros(multiple_tests_num)
        for k in range(1, multiple_tests_num + 1):
            alpha_hsidak[k - 1] = 1 - (1 - alpha) ** (1 / (multiple_tests_num + 1 - k))
        q = get_quantile(scores, alpha_hsidak, n)
        for s in range(nr_scores // multiple_tests_num):
            i, j = s * multiple_tests_num, s * multiple_tests_num + multiple_tests_num
            quant[i:j] = torch.diagonal(q[:, i:j])

    elif box_correction == "simes":
        # FWER Simes correction
        alpha_simes = torch.zeros(multiple_tests_num)
        for k in range(1, multiple_tests_num + 1):
            alpha_simes[k - 1] = k * alpha / multiple_tests_num
        assert torch.any(
            alpha.repeat(multiple_tests_num) - alpha_simes <= 0
        ), "H is not rejected"
        q = get_quantile(scores, alpha_simes, n)
        for s in range(nr_scores // multiple_tests_num):
            i, j = s * multiple_tests_num, s * multiple_tests_num + multiple_tests_num
            quant[i:j] = torch.diagonal(q[:, i:j])

    elif box_correction == "rank_global":
        # Max-rank algorithm v1: coordinate quantiles as informed via global quantile over max ranks
        ranks = torch.argsort(torch.argsort(scores, dim=0), dim=0)
        for s in range(nr_scores // multiple_tests_num):
            i, j = s * multiple_tests_num, s * multiple_tests_num + multiple_tests_num
            max_rank = torch.max(ranks[:, i:j], dim=1)[0]
            q_rank = get_quantile(max_rank.to(torch.float32), alpha, n)
            quant[i:j] = torch.sort(scores, dim=0)[0][int(q_rank), i:j]

    elif box_correction == "rank_coord":
        # Max-rank algorithm v2: coordinate quantiles as informed via global quantile over max ranks with coordinate-wise improvement
        # This is the method used in the paper and usually gives slightly tighter intervals than Max-rank algorithm v1
        ranks = torch.argsort(torch.argsort(scores, dim=0), dim=0)
        for s in range(nr_scores // multiple_tests_num):
            i, j = s * multiple_tests_num, s * multiple_tests_num + multiple_tests_num
            argsort_max_rank = torch.argsort(torch.max(ranks[:, i:j], dim=1)[0])
            q_rank = get_quantile(argsort_max_rank.to(torch.float32), alpha, n)
            incl_el = argsort_max_rank[: int(q_rank)]
            if incl_el.numel() == 0:  # incl_el is empty (degenerate case)
                incl_el = torch.tensor([0])
            q_rank_coord = torch.max(ranks[incl_el, i:j], dim=0)[0]
            quant[i:j] = torch.diagonal(torch.sort(scores, dim=0)[0][q_rank_coord, i:j])

    elif box_correction == "score_global":
        # Global quantile using max on scores directly instead of on ranks (as in Max-rank v1)
        # This correction is also considered by Andeol et al. (2023)
        for s in range(nr_scores // multiple_tests_num):
            i, j = s * multiple_tests_num, s * multiple_tests_num + multiple_tests_num
            max_score = torch.max(scores[:, i:j], dim=1)[0]
            quant[i:j] = get_quantile(max_score, alpha, n)

    else:  # no box correction
        quant[:] = get_quantile(scores, alpha, n)

    quant[0] = quant[0] * torch.pi  # max_angle  #
    quant[1] = quant[1] * (torch.sqrt(torch.tensor(2.0)))  # max_length  #
    # print("Scores after rescaling", quant)
    scores[:, 0] = scores[:, 0] * torch.pi  # max_angle  #
    scores[:, 1] = scores[:, 1] * (torch.sqrt(torch.tensor(2.0)))  # max_length  #
    return quant


def cp_compute_corrected_quantiles_and_coverage(
    test_conformity_angles,
    test_conformity_length,
    val_conformity_angles,
    val_conformity_length,
    q,
):
    """
    Computes joint quantiles and joint coverage using different correction methods.

    Args:
        test_abs_angles_errors (torch.Tensor): Absolute angle errors on the test set.
        test_abs_length_errors (torch.Tensor): Absolute length errors on the test set.
        val_abs_angles_errors (torch.Tensor): Absolute angle errors on the validation set.
        val_abs_length_errors (torch.Tensor): Absolute length errors on the validation set.
        quantiles_values (list of float): List of quantile values to compute.

    Returns:
        dict: A dictionary containing joint quantiles and joint coverage for each quantile and correction method.
    """
    angle_length_quantiles = {}
    joint_coverage_corrected = {}

    corrections = [
        # "naive_max",
        # "naive_mean",
        # "naive_quant",
        "bonferroni",
        "bonferroni_sidak",
        # "holm",
        # "holm_sidak",
        # "simes",
        "rank_global",
        # "rank_coord",
        "score_global",
    ]

    merged_val_errors = torch.cat(
        (val_conformity_angles, val_conformity_length), dim=-1
    )
    merged_test_errors = torch.cat(
        (test_conformity_angles, test_conformity_length), dim=-1
    )
    # print(merged_val_errors.shape, merged_test_errors.shape)
    alpha = 1.0 - q
    plot_histograms_angles_length(
        merged_val_errors,
        "/plots/quantiles_CP.png",
    )

    for correction in corrections:
        angle_length_quantiles[correction] = compute_quantile_correct(
            merged_val_errors,
            correction,
            2,
            merged_val_errors.shape[0],
            torch.tensor(alpha),
        )

        joint_coverage_corrected[correction] = (
            (merged_test_errors < angle_length_quantiles[correction])
            .all(dim=-1)
            .float()
            .mean()
        )
        angle_length_quantiles[correction] = (
            angle_length_quantiles[correction] * 2
        )  # it's the interval as it is double

    return {
        "intervals_width": angle_length_quantiles,
        "joint_coverage": joint_coverage_corrected,
        "angle_quantile_cp": angle_length_quantiles["bonferroni_sidak"][0] / 2,
        "length_quantile_cp": angle_length_quantiles["bonferroni_sidak"][1] / 2,
    }


def cqr_compute_corrected_quantiles_and_coverage(
    test_targets,
    all_predictions_test,
    val_conformity_angles,
    val_conformity_length,
    all_gt_vector_angles_test,
    all_pred_vector_angles_test,
    q,
):

    angle_length_quantiles = {}
    joint_coverage_corrected = {}
    interval_widths_angle_length = {}

    corrections = [
        # "naive_max",
        # "naive_mean",
        # "naive_quant",
        "bonferroni",
        "bonferroni_sidak",
        # "holm",
        # "holm_sidak",
        # "simes",
        "rank_global",
        # "rank_coord",
        "score_global",
    ]

    merged_val_scores = torch.cat(
        (val_conformity_angles.unsqueeze(-1), val_conformity_length.unsqueeze(-1)),
        dim=-1,
    )

    plot_histograms_angles_length(
        merged_val_scores,
        "/plots/quantiles_CQR.png",
    )
    plot_histograms_angles_length(
        torch.tensor(test_targets),
        "/plots/targets_distribution_test.png",
    )
    alpha = 1.0 - q

    for correction in corrections:
        # quantile corrected to be summed to the prediction
        angle_length_quantiles[correction] = compute_quantile_correct(
            merged_val_scores,
            correction,
            2,
            merged_val_scores.shape[0],
            torch.tensor(alpha),
        )

        lower_bound_cqr_angles = (
            all_predictions_test[:, 0, 0] - angle_length_quantiles[correction][0].item()
        )  # + all_pred_vector_angles_test
        upper_bound_cqr_angles = (
            all_predictions_test[:, 0, 1] + angle_length_quantiles[correction][0].item()
        )  # + all_pred_vector_angles_test

        lower_bound_cqr_lengths = (
            all_predictions_test[:, 1, 0] - angle_length_quantiles[correction][1].item()
        )
        upper_bound_cqr_lengths = (
            all_predictions_test[:, 1, 1] + angle_length_quantiles[correction][1].item()
        )

        joint_coverage_corrected[correction] = torch.mean(
            torch.tensor(
                (
                    (all_gt_vector_angles_test - all_pred_vector_angles_test + np.pi)
                    % (2 * np.pi)
                    - np.pi
                    > lower_bound_cqr_angles
                )
                & (
                    (all_gt_vector_angles_test - all_pred_vector_angles_test + np.pi)
                    % (2 * np.pi)
                    - np.pi
                    < upper_bound_cqr_angles
                )
                & (test_targets[:, 1] > lower_bound_cqr_lengths)
                & (test_targets[:, 1] < upper_bound_cqr_lengths)
            ).float()
        ).item()

        intervals_width_cqr_angles = torch.tensor(
            upper_bound_cqr_angles - lower_bound_cqr_angles
        )
        mean_interval_width_angle_cqr = torch.mean(intervals_width_cqr_angles).item()

        intervals_width_cqr_lengths = torch.tensor(
            upper_bound_cqr_lengths - lower_bound_cqr_lengths
        )
        mean_interval_width_length_cqr = torch.mean(intervals_width_cqr_lengths).item()
        interval_widths_angle_length[correction] = (
            mean_interval_width_angle_cqr,
            mean_interval_width_length_cqr,
        )
        if correction == "bonferroni_sidak":
            lower_bound_cqr_angles_corr = lower_bound_cqr_angles
            upper_bound_cqr_angles_corr = upper_bound_cqr_angles
            lower_bound_cqr_lengths_corr = lower_bound_cqr_lengths
            upper_bound_cqr_lengths_corr = upper_bound_cqr_lengths

    return {
        "joint_coverage": joint_coverage_corrected,
        "intervals_width": interval_widths_angle_length,
        "lower_bound_cqr_angles": lower_bound_cqr_angles_corr,
        "upper_bound_cqr_angles": upper_bound_cqr_angles_corr,
        "lower_bound_cqr_lengths": lower_bound_cqr_lengths_corr,
        "upper_bound_cqr_lengths": upper_bound_cqr_lengths_corr,
    }


if __name__ == "__main__":
    # Path to the HDF5 file
    val_file_path = "val_dataset_file.hdf5"
    test_file_path = "test_dataset_file.hdf5"
    set_seed(42)

    # Dataset parameters
    threshold_low = 0.1
    threshold_high = 3
    num_forecast_frames = 8
    quantiles_values = [0.7]

    val_dataset = HDF5TorchDataset(
        val_file_path,
    )

    val_data_loader = DataLoader(
        val_dataset, batch_size=2048, shuffle=True, num_workers=4
    )

    test_dataset = HDF5TorchDataset(
        test_file_path,
    )

    test_data_loader = DataLoader(
        test_dataset, batch_size=2048, shuffle=True, num_workers=4
    )

    # Conformal prediction
    test_abs_angles_errors = compute_abs_angles_errors(test_data_loader)
    test_abs_length_errors = compute_abs_length_errors(test_data_loader)
    val_abs_angles_errors = compute_abs_angles_errors(val_data_loader)
    val_abs_length_errors = compute_abs_length_errors(val_data_loader)

    # print(
    #     "Absolute Errors (sin, cos):", abs_errors_sin_cos[:5]
    # )  # Print first 5 for brevity
    # print(
    #     "Absolute Angles Errors (radians):", abs_angles_errors[:5]
    # )  # Print first 5 for brevity
    # Compute quantiles for angles
    angle_quantiles = {}
    length_quantiles = {}
    angle_length_quantiles = {}
    joint_coverage_corrected = {}
    corrections = [
        "naive_max",
        "naive_mean",
        "naive_quant",
        "bonferroni",
        "bonferroni_sidak",
        "holm",
        "holm_sidak",
        "simes",
        "rank_global",
        "rank_coord",
        "score_global",
    ]

    for q in quantiles_values:
        alpha = 1.0 - q
        angle_quantiles[q] = get_quantile(
            val_abs_angles_errors,
            torch.tensor(1.0 - q),
            val_abs_angles_errors.shape[0],
        )
        length_quantiles[q] = get_quantile(
            val_abs_length_errors,
            torch.tensor(1.0 - q),
            val_abs_length_errors.shape[0],
        )
        # CP coverage and interval size
        coverage_angle = (
            (test_abs_angles_errors < angle_quantiles[q].item()).float().mean()
        )
        coverage_length = (
            (test_abs_length_errors < length_quantiles[q].item()).float().mean()
        )
        joint_coverage = (
            (
                (test_abs_angles_errors < angle_quantiles[q].item())
                & (test_abs_length_errors < length_quantiles[q].item())
            )
            .float()
            .mean()
        )
        print(f"Coverage of classic CP for angles: {coverage_angle * 100:.2f}%")
        print(f"Coverage of classic CP for length: {coverage_length * 100:.2f}%")
        print(f"Joint coverage with no correction: {joint_coverage * 100:.2f}")
        print(
            f"Interval size of classic CP for angles:  {(angle_quantiles[q].item())*2:.4f} rad. ({(angle_quantiles[q]*2/torch.pi).item():.4f} π)"
        )
        print(
            f"Interval size of classic CP for length:  {(length_quantiles[q].item())*2:.4f}. (Yolo style)"
        )
        print("#####################################")
        merged_scores = torch.cat(
            (test_abs_angles_errors, test_abs_length_errors), dim=-1
        )
        # print(test_abs_angles_errors.shape, test_abs_length_errors.shape, merged_scores.shape)
        angle_length_quantiles[q] = {}
        joint_coverage_corrected[q] = {}
        for correction in corrections:
            angle_length_quantiles[q][correction] = compute_quantile_correct(
                merged_scores,
                correction,
                2,
                merged_scores.shape[0],
                torch.tensor(1.0 - q),
            )
            print(f"  {correction}: {angle_length_quantiles[q][correction]}")
            joint_coverage_corrected[q][correction] = (
                (
                    (
                        test_abs_angles_errors
                        < angle_length_quantiles[q][correction][0].item()
                    )
                    & (
                        test_abs_length_errors
                        < angle_length_quantiles[q][correction][1].item()
                    )
                )
                .float()
                .mean()
            )
            print(
                f"  {correction}: has joint coverage {joint_coverage_corrected[q][correction]}"
            )

    # for q in quantiles_values:
    #     print(f"Quantile: {q}")
    #     print(f"Angle Quantile: {angle_quantiles[q]}")
    #     print("Sin/Cos Corrected Quantiles:")
    #     for correction in corrections:
    #         print(f"  {correction}: {sin_cos_quantiles[q][correction]}")

# %%
