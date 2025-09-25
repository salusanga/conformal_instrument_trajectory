import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import h5py
import numpy as np
from torch.utils.data import DataLoader, Subset
import shutil
from data.datasethdf5_torch import HDF5TorchDataset


class HDF5SubsetDataset(HDF5TorchDataset):
    def __init__(self, original_dataset, selected_patients):
        """
        A subset of HDF5TorchDataset that contains only selected patients.

        Args:
            original_dataset (HDF5TorchDataset): The full dataset.
            selected_patients (list): List of patient IDs to include in the subset.
        """
        self.file = original_dataset.file  # Use the same HDF5 file
        self.patients = selected_patients  # Keep only selected patients
        self.num_forecast_frames = original_dataset.num_forecast_frames

        # Filter data to include only selected patients
        self.data = [
            (patient, timestamp)
            for patient, timestamp in original_dataset.data
            if patient in self.patients
        ]

    def get_patients(self):
        """Returns the list of patients in this subset."""
        return self.patients


# Function to split dataset
def split_dataset(dataset, split_ratio_val):
    """
    Splits the dataset into two HDF5SubsetDatasets containing a certain percentage of patients.

    Args:
        dataset (HDF5TorchDataset): The full dataset.
        split_ratio (float): The fraction of patients to include in the first split.
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of workers for loading data.

    Returns:
        DataLoader, DataLoader, HDF5SubsetDataset, HDF5SubsetDataset:
        Two DataLoaders and their corresponding dataset subsets.
    """
    # Get list of patients and shuffle
    patients = dataset.get_patients()
    # print(patients, len(patients))
    np.random.shuffle(patients)
    # Split into two groups
    split_idx = int(len(patients) * split_ratio_val)
    patients_val = patients[:split_idx]
    patients_test = patients[split_idx:]
    print(f"Patients val: {patients_val}, patients test: {patients_test}")

    # Create subset datasets
    subset_val = HDF5SubsetDataset(dataset, patients_val)
    subset_test = HDF5SubsetDataset(dataset, patients_test)

    # Create DataLoaders
    loader_val = DataLoader(subset_val, batch_size=8192, shuffle=False, num_workers=4)
    loader_test = DataLoader(subset_test, batch_size=8192, shuffle=False, num_workers=4)
    print(
        f"Total number of val sequences: {subset_val.get_total_sequences()} ({len(patients_val)} patients). Tot number of test sequences {subset_test.get_total_sequences()} ({len(patients_test)} patients)."
    )

    return loader_val, loader_test, subset_val, subset_test


def merge_hdf5_files(file1, file2, output_file):
    """
    Merges two HDF5 files that follow the structure used in HDF5TorchDataset.

    Args:
        file1 (str): Path to the first HDF5 file.
        file2 (str): Path to the second HDF5 file.
        output_file (str): Path to the merged output HDF5 file.

    The function ensures that patients from both files are copied to the new file without duplication.
    """
    # Copy the first file to the output file
    shutil.copy(file1, output_file)

    # Open the copied file in append mode and the second file in read mode
    with h5py.File(output_file, "a") as out_f, h5py.File(file2, "r") as f2:
        for patient in f2.keys():
            if patient in out_f:
                print(
                    f"Warning: Patient {patient} already exists in {output_file}. Skipping."
                )
                continue  # Skip patients that already exist

            # Copy the entire patient group
            f2.copy(patient, out_f)

    print(f"Merge completed. Output saved to {output_file}")


class HDF5Dataset:
    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")
        self.patients = list(self.file.keys())

    def get_patients(self):
        return list(self.patients)

    def get_timestamps(self, patient):
        timestamps = list(self.file[patient].keys())
        timestamps.sort(key=int)
        return timestamps

    def get_data(self, patient, timestamp):
        return self.file[patient][timestamp]

    def get_filtered_timestamps(self, patient, threshold_low, threshold_high=2):
        """
        Return timestamps where threshold_low < sqrt(gt_deltax^2 + gt_deltay^2) <= threshold_high.

        Args:
        - patient (str): Patient ID.
        - threshold_low (float): Lower bound of the distance threshold.
        - threshold_high (float): Upper bound of the distance threshold.

        Returns:
        - List[str]: Filtered list of timestamps.
        """
        filtered_timestamps = []
        for timestamp in self.get_timestamps(patient):
            data = self.get_data(patient, timestamp)
            gt_deltax = data["gt_deltax"][:][0]  # Extract scalar value
            gt_deltay = data["gt_deltay"][:][0]  # Extract scalar value
            distance = np.sqrt(gt_deltax**2 + gt_deltay**2)
            if threshold_low < distance <= threshold_high:
                filtered_timestamps.append(timestamp)
        return filtered_timestamps

    def filter_timestamps_with_internal_zeros_instrument(self, patient, timestamps):
        """
        Filters out timestamps where the `future_instr` array associated with a timestamp
        has at least one zero in column 0, excluding the first and last elements.

        Args:
        - dataset (HDF5Dataset): The dataset object for accessing data.
        - patient (str): Patient ID.
        - timestamps (list): List of timestamps to filter.

        Returns:
        - list: Filtered list of timestamps.
        """
        filtered_timestamps = []

        for timestamp in timestamps:
            data = self.get_data(patient, timestamp)
            future_instr = data["future_instr"][:]

            # Extract column 0 from future_instr
            col_0 = future_instr[:, 0]

            # Check for zeros excluding the first and last positions
            if np.any(col_0[1:-1] == 0):  # Check internal elements
                continue  # Skip this timestamp if any internal zero is found

            # Add timestamp to the filtered list if no zeros are found in the middle
            filtered_timestamps.append(timestamp)

        return filtered_timestamps

    def get_final_filtered_timestamps(
        self, patient, threshold_low, threshold_high, num_forecast_frames
    ):
        """
        Returns the final filtered timestamps for a patient.
        Args:
        - patient (str): Patient ID.
        - threshold_low (float): Lower bound of the distance threshold.
        - threshold_high (float): Upper bound of the distance threshold.
        - num_forecast_frames (int): Number of forecast frames.
        Returns:
        - List[str]: Final filtered list of timestamps.
        """
        step = 64 + num_forecast_frames
        # Get filtered timestamps
        filtered_timestamps = self.get_filtered_timestamps(
            patient, threshold_low, threshold_high
        )
        # print(f"Number of filtered timestamps: {len(filtered_timestamps)}")
        filtered_timestamps_zeros = (
            self.filter_timestamps_with_internal_zeros_instrument(
                patient, filtered_timestamps
            )
        )
        return filtered_timestamps_zeros


def save_filtered_hdf5(
    input_file_path,
    output_file_path,
    threshold_low,
    threshold_high,
    num_forecast_frames,
):
    """
    Saves a new HDF5 dataset with filtered timestamps for each patient.

    Args:
    - input_file_path (str): Path to the original HDF5 file.
    - output_file_path (str): Path to save the filtered HDF5 file.
    - threshold_low (float): Lower bound of the distance threshold.
    - threshold_high (float): Upper bound of the distance threshold.
    - num_forecast_frames (int): Number of forecast frames.
    """
    # Load the original dataset
    dataset = HDF5Dataset(input_file_path)

    # Create a new HDF5 file for the filtered dataset
    with h5py.File(output_file_path, "w") as new_file:
        # Iterate through each patient
        for patient in dataset.get_patients():
            # Get filtered timestamps
            filtered_timestamps = dataset.get_final_filtered_timestamps(
                patient, threshold_low, threshold_high, num_forecast_frames
            )

            # Create a group for the patient in the new file
            patient_group = new_file.create_group(patient)

            # Copy the data for each filtered timestamp
            for timestamp in filtered_timestamps:
                timestamp_data = dataset.get_data(patient, timestamp)

                # Create a group for the timestamp in the patient's group
                timestamp_group = patient_group.create_group(timestamp)

                # Copy datasets in the timestamp group
                for key, value in timestamp_data.items():
                    # Write each dataset (assume all values are numpy arrays)
                    timestamp_group.create_dataset(key, data=value[:])

    print(f"Filtered HDF5 file saved to: {output_file_path}")
