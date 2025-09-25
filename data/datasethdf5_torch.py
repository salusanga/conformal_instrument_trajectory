import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from utils.set_seed import set_seed


class HDF5TorchDataset(Dataset):
    def __init__(self, file_path, num_forecast_frames=8):
        """
        PyTorch Dataset for HDF5 data.

        Args:
        - file_path (str): Path to the HDF5 file.
        - num_forecast_frames (int): Number of forecast frames.
        """
        self.file = h5py.File(file_path, "r")
        self.patients = list(self.file.keys())
        self.num_forecast_frames = num_forecast_frames

        # Precompute the filtered timestamps for all patients
        self.data = []
        for patient in self.patients:
            filtered_timestamps = self.get_timestamps(
                patient
            )  # Assume filtering already happened
            for timestamp in filtered_timestamps:
                self.data.append(
                    (patient, timestamp)
                )  # Store (patient, timestamp) pairs

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)
    def get_item_by_patient_timestamp(self, patient, timestamp):
            """
            Retrieves the features, target, and additional required fields for a specific patient and timestamp.
            """
            data = self.get_data(patient, timestamp)

            # Extract features
            features = data["features"][:].astype(np.float32)  # (16,)

            # Ground truth and predicted vectors
            gt_deltax = sum(data["gt_changes"][-self.num_forecast_frames :, 1])
            gt_deltay = sum(data["gt_changes"][-self.num_forecast_frames :, 2])
            pred_deltax = sum(data["pred_changes"][-self.num_forecast_frames :, 1])
            pred_deltay = sum(data["pred_changes"][-self.num_forecast_frames :, 2])

            # Ground truth and predicted angles
            gt_angle = np.arctan2(gt_deltay, gt_deltax).astype(np.float32)
            pred_angle = np.arctan2(pred_deltay, pred_deltax).astype(np.float32)

            # Additional fields
            future_instr = data["future_instr"][:]  # (8, 5)
            gt_changes = data["gt_changes"][:]  # (8, 5)
            pred_changes = data["pred_changes"][:]  # (8, 5)
            pred_future_instr = data["pred_future_instr"][:]  # (8, 5)

            return (
                {
                    "features": torch.tensor(features).squeeze(),  # (16,)
                    "gt_angle": torch.tensor(gt_angle),  # Scalar
                    "gt_deltax": torch.tensor(gt_deltax),  # Scalar
                    "gt_deltay": torch.tensor(gt_deltay),  # Scalar
                    "gt_future_instr": torch.tensor(future_instr),  # (8, 5)
                    "gt_changes": torch.tensor(gt_changes),  # (8, 5)
                    "pred_angle": torch.tensor(pred_angle),  # Scalar
                    "pred_deltax": torch.tensor(pred_deltax),  # Scalar
                    "pred_deltay": torch.tensor(pred_deltay),  # Scalar
                    "pred_future_instr": torch.tensor(pred_future_instr),  # (8, 5)
                    "pred_changes": torch.tensor(pred_changes),  # (8, 5)
                },
                patient,
                timestamp,
            )
    def __getitem__(self, idx):
        """
        Retrieves the features, target, and additional required fields for a specific index.
        """
        patient, timestamp = self.data[idx]
        data = self.get_data(patient, timestamp)

        # Extract features
        features = data["features"][:].astype(np.float32)  # (16,)

        # Ground truth and predicted vectors
        gt_deltax = sum(data["gt_changes"][-self.num_forecast_frames :, 1])
        gt_deltay = sum(data["gt_changes"][-self.num_forecast_frames :, 2])
        pred_deltax = sum(data["pred_changes"][-self.num_forecast_frames :, 1])
        pred_deltay = sum(data["pred_changes"][-self.num_forecast_frames :, 2])

        # Ground truth and predicted angles
        gt_angle = np.arctan2(gt_deltay, gt_deltax).astype(np.float32)
        pred_angle = np.arctan2(pred_deltay, pred_deltax).astype(np.float32)

        # Additional fields
        future_instr = data["future_instr"][:]  # (8, 5)
        gt_changes = data["gt_changes"][:]  # (8, 5)
        pred_changes = data["pred_changes"][:]  # (8, 5)
        pred_future_instr = data["pred_future_instr"][:]  # (8, 5)

        return (
            {
                "features": torch.tensor(features).squeeze(),  # (16,)
                "gt_angle": torch.tensor(gt_angle),  # Scalar
                "gt_deltax": torch.tensor(gt_deltax),  # Scalar
                "gt_deltay": torch.tensor(gt_deltay),  # Scalar
                "gt_future_instr": torch.tensor(future_instr),  # (8, 5)
                "gt_changes": torch.tensor(gt_changes),  # (8, 5)
                "pred_angle": torch.tensor(pred_angle),  # Scalar
                "pred_deltax": torch.tensor(pred_deltax),  # Scalar
                "pred_deltay": torch.tensor(pred_deltay),  # Scalar
                "pred_future_instr": torch.tensor(pred_future_instr),  # (8, 5)
                "pred_changes": torch.tensor(pred_changes),  # (8, 5)
            },
            patient,
            timestamp,
        )

    def get_patients(self):
        return list(self.patients)

    def get_timestamps(self, patient):
        timestamps = list(self.file[patient].keys())
        timestamps.sort(key=int)
        return timestamps

    def get_data(self, patient, timestamp):
        return self.file[patient][timestamp]

    def get_total_sequences(self):
        return len(self.data)

    def get_filtered_timestamps(self, patient, threshold_low, threshold_high=2):
        filtered_timestamps = []
        for timestamp in self.get_timestamps(patient):
            data = self.get_data(patient, timestamp)
            gt_deltax = data["gt_deltax"][:][0]
            gt_deltay = data["gt_deltay"][:][0]
            distance = np.sqrt(gt_deltax**2 + gt_deltay**2)
            if threshold_low < distance <= threshold_high:
                filtered_timestamps.append(timestamp)
        return filtered_timestamps

    def filter_timestamps_with_internal_zeros_instrument(self, patient, timestamps):
        filtered_timestamps = []
        for timestamp in timestamps:
            data = self.get_data(patient, timestamp)
            future_instr = data["future_instr"][:]
            col_0 = future_instr[:, 0]
            if np.any(col_0[1:-1] == 0):
                continue
            filtered_timestamps.append(timestamp)
        return filtered_timestamps

    def get_final_filtered_timestamps(
        self, patient, threshold_low, threshold_high, num_forecast_frames
    ):
        step = 64 + num_forecast_frames
        filtered_timestamps = self.get_filtered_timestamps(
            patient, threshold_low, threshold_high
        )
        filtered_timestamps_zeros = (
            self.filter_timestamps_with_internal_zeros_instrument(
                patient, filtered_timestamps
            )
        )
        sampled_timestamps = []
        last_timestamp = None
        for timestamp in filtered_timestamps_zeros:
            if last_timestamp is None or int(timestamp) - int(last_timestamp) >= step:
                sampled_timestamps.append(timestamp)
                last_timestamp = timestamp
        return sampled_timestamps


if __name__ == "__main__":
    # Path to the HDF5 file
    file_path = (
        "dataset_file.hdf5"
    )
    set_seed(42)

    # Dataset parameters
    threshold_low = 0.1
    threshold_high = 3
    num_forecast_frames = 8

    # Initialize the dataset
    dataset = HDF5TorchDataset(file_path, num_forecast_frames)

    # Create DataLoader
    batch_size = 32
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Example usage
    for batch_data, patient, timestamp in data_loader:
        # Access the dictionary from batch_data
        batch_features = batch_data["features"]
        batch_angles_gt = batch_data["gt_angle"]
        batch_dx_gt = batch_data["gt_deltax"]
        batch_dy_gt = batch_data["gt_deltay"]
        future_instr_gt = batch_data["gt_future_instr"]
        changes_gt = batch_data["gt_changes"]
        batch_angles_pred = batch_data["pred_angle"]
        batch_dx_pred = batch_data["pred_deltax"]
        batch_dy_pred = batch_data["pred_deltay"]
        future_instr_pred = batch_data["pred_future_instr"]
        changes_pred = batch_data["pred_changes"]

        # Print the shapes of the batch data
        print("Features shape:", batch_features.shape)  # Shape: (batch_size, 16)
        print(
            "Ground truth angles shape:", batch_angles_gt.shape
        )  # Shape: (batch_size,)
        print("Ground truth dx shape:", batch_dx_gt.shape)  # Shape: (batch_size,)
        print("Ground truth dy shape:", batch_dy_gt.shape)  # Shape: (batch_size,)
        print("Future instructions shape:", future_instr_gt.shape)
        print("Ground truth changes shape:", changes_gt.shape)
        print("Predicted angles shape:", batch_angles_pred.shape)
        print("Predicted dx shape:", batch_dx_pred.shape)
        print("Predicted dy shape:", batch_dy_pred.shape)
        print("Patient ID:", patient)
        print("Timestamp:", timestamp)

        break
