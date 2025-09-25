# Conformal Forecasting for Surgical Instrument Trajectory 🎯

This repository contains the code and resources for the MICCAI 2025 paper titled **"Conformal forecasting for surgical instrument trajectory"**. The work explores uncertainty estimation in forecasting surgical instrument motion, leveraging conformal prediction techniques.

## Paper Link 📄
[Conformal forecasting for surgical instrument trajectory (MICCAI 2025)](https://papers.miccai.org/miccai-2025/paper/0260_paper.pdf)

## Abstract
Forecasting surgical instrument trajectories and predicting the next surgical action recently started to attract attention from the research community. Both these tasks are crucial for automation and assistance in endoscopy surgery. Given the safety-critical nature of these tasks, reliable uncertainty quantification is essential.

Conformal prediction is a fast-growing and widely recognized framework for uncertainty estimation in machine learning and computer vision, offering distribution-free, theoretically valid prediction intervals. In this work, we explore the application of standard conformal prediction and conformalized quantile regression to estimate uncertainty in forecasting surgical instrument motion, i.e., predicting the direction and magnitude of surgical instruments’ future motion.

We analyze and compare their coverage and interval sizes, assessing the impact of multiple hypothesis testing and correction methods. Additionally, we show how these techniques can be employed to produce useful uncertainty heatmaps. To the best of our knowledge, this is the first study applying conformal prediction to surgical guidance, marking an initial step toward constructing principled prediction intervals with formal coverage guarantees in this domain.

---

## Citation
Please cite this work as:
```bibtex
@InProceedings{SanSar_Conformal_MICCAI2025,
  author = {Sangalli, Sara AND Sarwin, Gary AND Erdil, Ertunc AND Serra, Carlo AND Carretta, Alessandro AND Staartjes, Victor AND Konukoglu, Ender},
  title = {Conformal forecasting for surgical instrument trajectory},
  booktitle = {Proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
  year = {2025},
  publisher = {Springer Nature Switzerland},
  volume = {LNCS 15968},
  month = {September},
  pages = {117--127}
}
```

---

## Dataset
The in-house dataset used in this work (plase refer to the paper for a description) is stored in an HDF5 file, organized hierarchically with each patient as a top-level group. Each patient group contains multiple timestamps, each representing a data sample.

#### Data Structure:
- **Root level:** Contains groups named after each patient (e.g., `"patient_1"`, `"patient_2"`, etc.).
- **Within each patient group:** Each timestamp is a subgroup (e.g., `"0001"`, `"0002"`, etc.).
- **Within each timestamp subgroup:** Several datasets storing different data components:
  - `"features"`: A float array of shape `(16,)`, representing extracted input features.
  - `"gt_deltax"`: Ground truth change in x-direction, shape `(1,)`.
  - `"gt_deltay"`: Ground truth change in y-direction, shape `(1,)`.
  - `"future_instr"`: Future instruction data, shape `(8, 5)`.
  - `"gt_changes"`: Ground truth changes, shape `(8, 5)`.
  - `"pred_changes"`: Predicted changes, shape `(8, 5)`.
  - `"pred_future_instr"`: Predicted future instructions, shape `(8, 5)`.

#### Data Sample:
Each sample retrieved from the dataset includes:
- `features`: Input features for the model, shape `(16,)`.
- `gt_angle`: Ground truth angle in radians (scalar).
- `gt_deltax`: Ground truth change in x (scalar).
- `gt_deltay`: Ground truth change in y (scalar).
- `gt_future_instr`: Future instruction data, shape `(8, 5)`.
- `gt_changes`: Ground truth change vectors, shape `(8, 5)`.
- `pred_angle`: Predicted angle in radians (scalar).
- `pred_deltax`: Predicted change in x (scalar).
- `pred_deltay`: Predicted change in y (scalar).
- `pred_future_instr`: Predicted future instructions, shape `(8, 5)`.
- `pred_changes`: Predicted change vectors, shape `(8, 5)`.

### Notes:
- Timestamps are sorted numerically.
- The dataset may include filtering based on distance thresholds or internal zeros in the instrument, as per your filtering functions.
- Ensure the HDF5 file structure adheres to this organization for compatibility with the provided DataLoader.\
---

## Usage

### Training conformal quantile regression
If you want to train a CQR network with overall target coverage 0.9, reading the input features extracted from model [1] used in this paper as input for the CQR network, and using the model config from this paper (which should be saved in the folder `/configs/cqra_training/`), for regressing quantiles for both angle and length, run:

```bash
python cqr_training.py --model_cqra_cfg cqra1_001.yaml --train_file_path train_hdf5_path.hdf5 --val_file_path val_hdf5_path.hdf5 --checkpoint_dir /checkpoint_folder/ --quantile 0.9 --test_type "angles_and_length"
```

### Evaluation of Methods
To run the evaluation of the different methods—conformal prediction, conformalized quantile regression, and multiple testing corrections (when applying conformalization to both angles and lengths)—using the input features extracted from model [1] (train and validation set split as explained in the paper), run:

```bash
python script_conformal_procedure.py --test_file_path path_to_your_eval_set.hdf5 --checkpoint_dir /checkpoint_folder/ --test_type "angles_and_length" --quantile 0.9
```

[1]: Sarwin, G., Carretta, A., Staartjes, V., Zoli, M., Mazzatenta, D., Regli, L., Serra, C., Konukoglu, E.: *Anatomy might be all you need: Forecasting what to do during surgery* (2025).

---

## Acknowledgments
This study was financially supported by: 1. The LOOP Zürich– Medical Research Center, Zurich, Switzerland, 2. Personalized Health and Related Technologies (PHRT), project number 222, ETH domain, 3. Clinical Research Priority Program (CRPP) Grant on Artificial Intelligence in Oncological Imaging Network, University of Zürich, 4. The SNSF (Project IZKSZ3_218786).
