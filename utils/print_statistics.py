import torch
import wandb


# Commenting out the parts concerning cqr_CORRECTED_overall_list
def print_statistics(
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
):
    """
    Prints the mean and standard deviation of CP and CQR coverage and interval width lists.

    Args:
        Lists of values (list or torch.Tensor).
    """

    def compute_stats(values):
        values = (
            torch.tensor(values, dtype=torch.float32) if values else torch.tensor([0.0])
        )
        return values.mean().item(), values.std().item()

    # Compute mean and std for all lists
    test_abs_angle_errors_mean, test_abs_angle_errors_std = compute_stats(
        test_abs_angles_errors_list
    )
    test_abs_length_errors_mean, test_abs_length_errors_std = compute_stats(
        test_abs_length_errors_list
    )
    test_length_gt_avg_mean, test_length_gt_avg_std = compute_stats(
        test_length_gt_avg_list
    )
    cp_cov_angle_mean, cp_cov_angle_std = compute_stats(cp_coverage_angle_list)
    cp_cov_length_mean, cp_cov_length_std = compute_stats(cp_coverage_length_list)
    cp_width_angle_mean, cp_width_angle_std = compute_stats(cp_int_width_angle_list)
    cp_width_length_mean, cp_width_length_std = compute_stats(cp_int_width_length_list)

    cqr_cov_angle_mean, cqr_cov_angle_std = compute_stats(cqr_coverage_angle_list)
    cqr_cov_length_mean, cqr_cov_length_std = compute_stats(cqr_coverage_length_list)
    cqr_width_angle_mean, cqr_width_angle_std = compute_stats(cqr_int_width_angle_list)
    cqr_width_length_mean, cqr_width_length_std = compute_stats(
        cqr_int_width_length_list
    )

    # Compute mean and std for non-corrected CP
    cp_non_corrected_coverage_mean, cp_non_corrected_coverage_std = compute_stats(
        cp_NON_corrected_overall_coverage_list
    )
    cp_non_corrected_width_mean, cp_non_corrected_width_std = compute_stats(
        cp_NON_corrected_overall_width_list
    )

    # Compute statistics for corrected CP methods
    cp_corrections = (
        cp_CORRECTED_overall_list[0]["joint_coverage"].keys()
        if cp_CORRECTED_overall_list
        else []
    )

    cp_corrected_coverage_stats = {correction: [] for correction in cp_corrections}
    cp_corrected_width_stats = {correction: [] for correction in cp_corrections}
    cp_corrected_width_stats_angle = {correction: [] for correction in cp_corrections}
    cp_corrected_width_stats_length = {correction: [] for correction in cp_corrections}

    for entry in cp_CORRECTED_overall_list:
        for correction in cp_corrections:
            cp_corrected_coverage_stats[correction].append(
                entry["joint_coverage"][correction]
            )
            cp_corrected_width_stats[correction].append(
                sum(entry["intervals_width"][correction]) / 2
            )
            cp_corrected_width_stats_angle[correction].append(
                entry["intervals_width"][correction][0]
            )
            cp_corrected_width_stats_length[correction].append(
                entry["intervals_width"][correction][1]
            )

    cp_corrected_coverage_means_stds = {
        corr: compute_stats(cp_corrected_coverage_stats[corr])
        for corr in cp_corrections
    }
    cp_corrected_width_means_stds = {
        corr: compute_stats(cp_corrected_width_stats[corr]) for corr in cp_corrections
    }
    cp_corrected_width_angle_means_stds = {
        corr: compute_stats(cp_corrected_width_stats_angle[corr])
        for corr in cp_corrections
    }
    cp_corrected_width_length_means_stds = {
        corr: compute_stats(cp_corrected_width_stats_length[corr])
        for corr in cp_corrections
    }

    # Compute mean and std for non-corrected CQR
    cqr_non_corrected_coverage_mean, cqr_non_corrected_coverage_std = compute_stats(
        cqr_NON_corrected_overall_coverage_list
    )
    cqr_non_corrected_width_mean, cqr_non_corrected_width_std = compute_stats(
        cqr_NON_corrected_overall_width_list
    )

    cqr_corrections = (
        cqr_CORRECTED_overall_list[0]["joint_coverage"].keys()
        if cqr_CORRECTED_overall_list
        else []
    )

    cqr_corrected_coverage_stats = {correction: [] for correction in cqr_corrections}
    cqr_corrected_width_stats = {correction: [] for correction in cqr_corrections}
    cqr_corrected_width_stats_angle = {correction: [] for correction in cqr_corrections}
    cqr_corrected_width_stats_length = {
        correction: [] for correction in cqr_corrections
    }

    for entry in cqr_CORRECTED_overall_list:
        for correction in cqr_corrections:
            cqr_corrected_coverage_stats[correction].append(
                entry["joint_coverage"][correction]
            )
            cqr_corrected_width_stats[correction].append(
                sum(entry["intervals_width"][correction]) / 2
            )
            cqr_corrected_width_stats_angle[correction].append(
                entry["intervals_width"][correction][0]
            )
            cqr_corrected_width_stats_length[correction].append(
                entry["intervals_width"][correction][1]
            )

    cqr_corrected_coverage_means_stds = {
        corr: compute_stats(cqr_corrected_coverage_stats[corr])
        for corr in cqr_corrections
    }
    cqr_corrected_width_means_stds = {
        corr: compute_stats(cqr_corrected_width_stats[corr]) for corr in cqr_corrections
    }
    cqr_corrected_width_angle_means_stds = {
        corr: compute_stats(cqr_corrected_width_stats_angle[corr])
        for corr in cqr_corrections
    }
    cqr_corrected_width_length_means_stds = {
        corr: compute_stats(cqr_corrected_width_stats_length[corr])
        for corr in cqr_corrections
    }

    print("#####################################")
    print("FINAL STATISTICS:")
    print("#####################################")
    print("TEST ERRORS")
    # Print test errors
    print(
        f"Mean absolute angle error: {(test_abs_angle_errors_mean/torch.pi)*180.0 :.2f}°  ({(test_abs_angle_errors_mean/torch.pi):.4f} π) ± {(test_abs_angle_errors_std/torch.pi)*180.0 :.2f}°"
    )
    print(
        f"Mean absolute length error: {test_abs_length_errors_mean:.4f} ± {test_abs_length_errors_std:.4f}"
    )
    print(
        f"Mean length ground truth: {test_length_gt_avg_mean:.4f} ± {test_length_gt_avg_std:.4f}"
    )
    print("CP")
    # Print classic CP statistics
    print(
        f"Coverage of classic CP for angles: {cp_cov_angle_mean * 100:.2f}% ± {cp_cov_angle_std * 100:.2f}%"
    )
    print(
        f"Coverage of classic CP for length: {cp_cov_length_mean * 100:.2f}% ± {cp_cov_length_std * 100:.2f}%"
    )
    print(
        f"Interval size of classic CP for angles: {(cp_width_angle_mean/torch.pi)*180.0 :.2f}° ({(cp_width_angle_mean/torch.pi):.4f} π) ± {(cp_width_angle_std/torch.pi)*180.0 :.2f}°"
    )
    print(
        f"Interval size of classic CP for length: {cp_width_length_mean:.4f} ± {cp_width_length_std:.4f} (Yolo style)"
    )
    print("CQR")
    # Print CQR statistics
    print(
        f"Coverage of CQR for angles: {cqr_cov_angle_mean * 100:.2f}% ± {cqr_cov_angle_std * 100:.2f}%"
    )
    print(
        f"Coverage of CQR for length: {cqr_cov_length_mean * 100:.2f}% ± {cqr_cov_length_std * 100:.2f}%"
    )
    print(
        f"Interval size of CQR for angles: {(cqr_width_angle_mean/torch.pi)*180.0 :.2f}° ({(cqr_width_angle_mean/torch.pi):.4f} π) ± {(cqr_width_angle_std/torch.pi)*180.0 :.2f}°"
    )
    print(
        f"Interval size of CQR for length: {cqr_width_length_mean*2:.4f} ± {cqr_width_length_std*2:.4f} (Yolo style)"
    )
    print("CP JOINT NON CORRECTED")
    # Print non-corrected CP statistics
    print("Coverage of non-corrected CP methods:")
    print(
        f"  Non-corrected CP: {cp_non_corrected_coverage_mean * 100:.2f}% ± {cp_non_corrected_coverage_std * 100:.2f}%"
    )
    print(
        f"  Interval size of non-corrected CP: {cp_non_corrected_width_mean:.4f} ± {cp_non_corrected_width_std:.4f}. This is the average of the two independent intervals."
    )
    print("CP JOINT CORRECTED")
    print("Coverage of CORRECTED CP methods:")
    for correction, (mean, std) in cp_corrected_coverage_means_stds.items():
        print(f"  {correction}: {mean * 100:.2f}% ± {std * 100:.2f}%")
    print("Avg. (angle and len) Interval size of CORRECTED CP methods:")
    for correction, (mean, std) in cp_corrected_width_means_stds.items():
        print(f"  {correction}: {mean:.4f} ± {std:.4f}")
    print("Angle Interval size of CORRECTED CP methods:")
    for correction, (mean, std) in cp_corrected_width_angle_means_stds.items():
        print(
            f"  {correction}: {(mean/torch.pi)*180.0 :.2f}° ± {(std/torch.pi)*180.0 :.2f}°"
        )
    print("Length Interval size of CORRECTED CP methods:")
    for correction, (mean, std) in cp_corrected_width_length_means_stds.items():
        print(f"  {correction}: {mean:.4f} ± {std:.4f}")

    print("CQR JOINT NON CORRECTED")
    # Print non-corrected CQR statistics
    print("Coverage of non-corrected CQR methods:")
    print(
        f"  Non-corrected CQR: {cqr_non_corrected_coverage_mean * 100:.2f}% ± {cqr_non_corrected_coverage_std * 100:.2f}%"
    )
    print(
        f"  Interval size of non-corrected CQR: {cqr_non_corrected_width_mean:.4f} ± {cqr_non_corrected_width_std:.4f}"
    )
    print("CQR JOINT CORRECTED")
    print("Coverage of CORRECTED CQR methods:")
    for correction, (mean, std) in cqr_corrected_coverage_means_stds.items():
        print(f"  {correction}: {mean * 100:.2f}% ± {std * 100:.2f}%")
    print("Avg. (angle and len) Interval size of CORRECTED CQR methods:")
    for correction, (mean, std) in cqr_corrected_width_means_stds.items():
        print(f"  {correction}: {mean:.4f} ± {std:.4f}")
    print("Angle Interval size of CORRECTED CQR methods:")
    for correction, (mean, std) in cqr_corrected_width_angle_means_stds.items():
        print(
            f"  {correction}: {(mean/torch.pi)*180.0 :.2f}° ± {(std/torch.pi)*180.0 :.2f}°"
        )
    print("Length Interval size of CORRECTED CQR methods:")
    for correction, (mean, std) in cqr_corrected_width_length_means_stds.items():
        print(f"  {correction}: {mean:.4f} ± {std:.4f}")

    # Log to wandb
    wandb.run.summary.update(
        {  # Test errors
            "test_abs_angle_errors_mean": test_abs_angle_errors_mean,
            "test_abs_angle_errors_std": test_abs_angle_errors_std,
            "test_abs_length_errors_mean": test_abs_length_errors_mean,
            "test_abs_length_errors_std": test_abs_length_errors_std,
            "test_length_gt_avg_mean": test_length_gt_avg_mean,
            "test_length_gt_avg_std": test_length_gt_avg_std,
            "cp_coverage_angle_mean": cp_cov_angle_mean,
            "cp_coverage_angle_std": cp_cov_angle_std,
            "cp_coverage_length_mean": cp_cov_length_mean,
            "cp_coverage_length_std": cp_cov_length_std,
            "cp_int_width_angle_mean": cp_width_angle_mean,
            "cp_int_width_angle_std": cp_width_angle_std,
            "cp_int_width_length_mean": cp_width_length_mean,
            "cp_int_width_length_std": cp_width_length_std,
            "cqr_coverage_angle_mean": cqr_cov_angle_mean,
            "cqr_coverage_angle_std": cqr_cov_angle_std,
            "cqr_int_width_angle_mean": cqr_width_angle_mean,
            "cqr_int_width_angle_std": cqr_width_angle_std,
            "cp_NON_corrected_overall_coverage_mean": cp_non_corrected_coverage_mean,
            "cp_NON_corrected_overall_coverage_std": cp_non_corrected_coverage_std,
            "cp_NON_corrected_overall_width_mean": cp_non_corrected_width_mean,
            "cp_NON_corrected_overall_width_std": cp_non_corrected_width_std,
            "cqr_NON_corrected_overall_coverage_mean": cqr_non_corrected_coverage_mean,
            "cqr_NON_corrected_overall_coverage_std": cqr_non_corrected_coverage_std,
            "cqr_NON_corrected_overall_width_mean": cqr_non_corrected_width_mean,
            "cqr_NON_corrected_overall_width_std": cqr_non_corrected_width_std,
            **{
                f"corrected_coverage_{correction}_mean": mean
                for correction, (mean, std) in cp_corrected_coverage_means_stds.items()
            },
            **{
                f"corrected_coverage_{correction}_std": std
                for correction, (mean, std) in cp_corrected_coverage_means_stds.items()
            },
            **{
                f"corrected_width_{correction}_mean": mean
                for correction, (mean, std) in cp_corrected_width_means_stds.items()
            },
            **{
                f"corrected_width_{correction}_std": std
                for correction, (mean, std) in cp_corrected_width_means_stds.items()
            },
            **{
                f"corrected_width_{correction}_mean": mean
                for correction, (
                    mean,
                    std,
                ) in cp_corrected_width_angle_means_stds.items()
            },
            **{
                f"corrected_width_{correction}_std": std
                for correction, (
                    mean,
                    std,
                ) in cp_corrected_width_angle_means_stds.items()
            },
            **{
                f"corrected_width_{correction}_mean": mean
                for correction, (
                    mean,
                    std,
                ) in cp_corrected_width_length_means_stds.items()
            },
            **{
                f"corrected_width_{correction}_std": std
                for correction, (
                    mean,
                    std,
                ) in cp_corrected_width_length_means_stds.items()
            },
            **{
                f"cqr_corrected_coverage_{correction}_mean": mean
                for correction, (mean, std) in cqr_corrected_coverage_means_stds.items()
            },
            **{
                f"cqr_corrected_coverage_{correction}_std": std
                for correction, (mean, std) in cqr_corrected_coverage_means_stds.items()
            },
            **{
                f"cqr_corrected_width_{correction}_mean": mean
                for correction, (mean, std) in cqr_corrected_width_means_stds.items()
            },
            **{
                f"cqr_corrected_width_{correction}_std": std
                for correction, (mean, std) in cqr_corrected_width_means_stds.items()
            },
            **{
                f"cqr_corrected_width_angle_{correction}_mean": mean
                for correction, (
                    mean,
                    std,
                ) in cqr_corrected_width_angle_means_stds.items()
            },
            **{
                f"cqr_corrected_width_angle_{correction}_std": std
                for correction, (
                    mean,
                    std,
                ) in cqr_corrected_width_angle_means_stds.items()
            },
            **{
                f"cqr_corrected_width_length_{correction}_mean": mean
                for correction, (
                    mean,
                    std,
                ) in cqr_corrected_width_length_means_stds.items()
            },
            **{
                f"cqr_corrected_width_length_{correction}_std": std
                for correction, (
                    mean,
                    std,
                ) in cqr_corrected_width_length_means_stds.items()
            },
        }
    )
