# MSAM: Magnification-adaptive Survival-aware Attention Model

[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--2640--3086-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/my-orcid?orcid=0009-0003-2640-3086)
[![GitHub](https://img.shields.io/badge/GitHub-minibelfast-181717?logo=github&logoColor=white)](https://github.com/minibelfast)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Xuanyu%20Wang-00CCBB?logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Xuanyu-Wang-11/research)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-xuanyuwang-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/xuanyuwang)

This repository contains the code for the **MSAM (Magnification-adaptive Survival-aware Attention Model)**. The model is built on top of the Mamba architecture and attention-based aggregation modules to achieve robust prognostic prediction, cross-cohort continual learning, and visualization from Whole Slide Images (WSIs).

## Project Structure

- `models/`: Contains the definition of MSAM-related models (e.g., `MambaAttn.py`, `MSA_MIL.py`) and other baseline MIL models.
- `part/`: Contains network components such as `TokenSelect`, `WTConv2d`, `GLSA`, `DFF`, etc.
- `mamba/`: Contains the core Selective Scan Space State Sequential Model (Mamba) implementation.
- `utils/`: Contains utilities for training, survival loss functions, evaluation, and continual learning (KD/EWC).
- `train_scripts/`: Bash scripts for running training pipelines.
- `splits/`: Train/val/test splits for cross-validation.
- `dataset/`: Dataset loaders (including multimodal survival dataset).
- `dataset_csv/`: Processed metadata tables (e.g., `*_processed.csv`) for survival prediction.
- `CLAM/`: WSI processing and visualization scripts (e.g., heatmaps, t-SNE, ERF).

## Installation

### Prerequisites
- CUDA 11.8
- Python 3.10

### Environment Setup
1. Create and activate a conda virtual environment:
   ```bash
   conda create -n gpma python=3.10 -y
   conda activate gpma
   ```
2. Install PyTorch 2.0.1:
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   pip install packaging
   ```
3. Install dependencies:
   ```bash
   pip install scikit-survival pandas h5py lifelines tqdm pyyaml timm scikit-learn matplotlib shap
   ```
4. Install Mamba kernel:
   ```bash
   pip install -e ./mamba
   ```

## Usage

### 1. Data Preparation
- Organize your WSIs and extract patch/slide features (feature-level training is used in this repository).
- For multimodal survival prediction, the dataset loader expects:
  - Titan features as `<data_root_dir_titan>/<slide_id>.h5` with dataset key `features`
  - UNI features as `<data_root_dir_uni>/<slide_id>.h5` with dataset key `features`
- Prepare a processed metadata CSV under `dataset_csv/` with columns:
  - `case_id`, `slide_id`, `censorship`, `survival_months`
- Prepare split CSVs under `splits/<YOUR_TASK>_kfold/` as `splits_0.csv ... splits_{k-1}.csv`

### 2. Training the MSAM Model (Survival Prediction)
We provide training scripts for survival prediction. To train the `mamba_attn` / MSAM-style survival model:

```bash
bash train_scripts/ATTN_512_survival_k_fold.sh
```

Alternatively, you can run the Python script directly:
```bash
python main_survival.py \
    --drop_out 0.3 \
    --early_stopping \
    --lr 2e-4 \
    --k 5 \
    --k_start 0 \
    --k_end 4 \
    --label_frac 1.0 \
    --max_epochs 200 \
    --model_type mamba_attn \
    --mambamil_layer 2 \
    --mambamil_rate 10 \
    --mambamil_type SRMamba \
    --opt adam \
    --reg 1e-5 \
    --bag_loss nll_surv \
    --task STAD_survival \
    --split_dir splits/STAD_survival_kfold \
    --data_root_dir_titan ./data/features_titan_h5 \
    --data_root_dir_uni ./data/features_uni_h5 \
    --results_dir ./results/STAD_survival_mamba_attn
```

### 3. Evaluation
To evaluate a saved checkpoint on a test set, use `model_eval.py`:
```bash
python model_eval.py
```
Note: you may need to edit the checkpoint path and arguments inside `model_eval.py` to point to your trained model.

### 4. Visualization
The `CLAM/` folder contains scripts to visualize the MSAM model's attention heatmaps and feature distributions:
- Heatmaps: `CLAM/create_heatmaps-tsne.py`, `CLAM/create_heatmaps.py`, `CLAM/create_heatmaps-UNI.py`
- t-SNE & ERF: `CLAM/create_tsne.py`, `CLAM/create_tsne_2D-new.py`, `CLAM/create_erf-mean.py`, `CLAM/create_erf-max.py`

Example command for heatmap generation:
```bash
python CLAM/create_heatmaps-tsne.py
```
Please refer to each script for its argument requirements and update paths accordingly.

## Pretrained Weights
You can host and organize the best-performing MSAM checkpoints using your Hugging Face page:
- https://huggingface.co/xuanyuwang

## Citation
If you use MSAM in your research, please cite your corresponding manuscript.

## Acknowledgements
This project is built upon https://github.com/isyangshu/MambaMIL and https://github.com/mahmoodlab/CLAM.
