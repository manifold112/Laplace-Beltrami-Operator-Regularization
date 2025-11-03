# Laplace-Beltrami Operator Regularization for Robust Skeleton-based Isolated Sign Language Recognition

This repository provides the official PyTorch implementation of our FG 2026 paper:

> **Laplace-Beltrami Operator Regularization for Robust Skeleton-based Isolated Sign Language Recognition**  
> IEEE International Conference on Automatic Face and Gesture Recognition (FG 2026)

LBOR is a plug-and-play training objective for skeleton-based isolated sign language recognition (ISLR). It regularizes the within-class feature geometry by building class-specific graphs in the embedding space and minimizing a Laplacian energy, while a lightweight center-level margin preserves between-class separation. The design is model-agnostic and can be attached to existing ISLR backbones without architectural changes.

---

## 1. Key Features

- **Laplace-Beltrami Operator Regularization (LBOR)**  
  Constructs within-class graphs in the feature space for each mini-batch and minimizes a Laplacian (Dirichlet) energy to enforce intra-class connectivity and smoothness, mitigating signer-driven multi-centroid fragmentation in ISLR.

- **Center-Level Margin Term**  
  Encourages a margin between class centers in the embedding space, preserving inter-class discriminability while LBOR regularizes each class manifold.

- **Model-Agnostic and Lightweight**  
  No change to backbone architectures (e.g. HMA, SignBERT-style models, SKIM). LBOR is implemented purely as an additional loss on top of standard classification losses and is compatible with any skeleton-based encoder that outputs per-instance embeddings.

- **Reproducible Evaluation on Public ISLR Benchmarks**  
  Experiments on word-level American Sign Language (WLASL) and Chinese Sign Language (NMFs-CSL) with unified training/evaluation scripts and configuration files.

---

## 2. Repository Structure

```text
Laplace-Beltrami-Operator-Regularization/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt            # or environment.yml
├── setup.py 
├── .gitignore

├── configs/                    # experiment configurations
│   ├── wlasl/
│   │   ├── hma_lbor.yaml
│   │   ├── signbert_lbor.yaml
│   │   └── skim_lbor.yaml
│   ├── nmfscsl/
│   │   ├── hma_lbor.yaml
│   │   └── skim_lbor.yaml
│   └── default.yaml

├── src/
│   └── lbor_islr/
│       ├── __init__.py
│       ├── models/
│       │   ├── hma.py
│       │   ├── signbert.py
│       │   ├── skim.py
│       │   └── builder.py
│       ├── losses/
│       │   ├── lbor_loss.py      # LBOR implementation
│       │   └── ce_variants.py
│       ├── datasets/
│       │   ├── wlasl.py
│       │   ├── nmfscsl.py
│       │   ├── transforms.py
│       │   └── utils.py
│       ├── engine/
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   └── scheduler.py
│       ├── utils/
│       │   ├── logger.py
│       │   ├── distributed.py
│       │   ├── seed.py
│       │   └── misc.py
│       ├── train.py              # training entry point
│       └── test.py               # evaluation entry point

├── scripts/
│   ├── train_wlasl_hma_lbor.sh
│   ├── train_wlasl_signbert_lbor.sh
│   ├── train_wlasl_skim_lbor.sh
│   ├── train_nmfscsl_hma_lbor.sh
│   └── eval_wlasl_hma_lbor.sh

├── tools/
│   ├── prepare_wlasl_skeleton.py
│   ├── prepare_nmfscsl_skeleton.py
│   └── visualize_skeleton.py

├── docs/
│   ├── INSTALL.md
│   ├── DATASETS.md
│   ├── EXPERIMENTS.md
│   ├── METHODS.md
│   └── FAQ.md

├── checkpoints/
│   └── README.md                 # links to pretrained weights (not stored in git)

└── figures/
    ├── method_overview.png
    ├── laplacian_graph.png
    └── center_margin.png
---

## 3. Installation

We recommend using a Conda environment.

### 3.1 Create environment

```bash
conda create -n lbor_islr python=3.10 -y
conda activate lbor_islr
```

### 3.2 Clone this repository

```bash
git clone <repository-url>
cd Laplace-Beltrami-Operator-Regularization
```

### 3.3 Install dependencies

```bash
pip install -r requirements.txt
```

The implementation has been tested with:

- Python 3.9 or later  
- PyTorch 1.12 or later (with a matching CUDA toolkit)  
- torchvision, PyYAML, NumPy, SciPy, tqdm and other standard libraries

---

## 4. Datasets

### 4.1 WLASL

We evaluate LBOR on WLASL (Word-Level American Sign Language) using 2D skeleton sequences extracted from RGB videos.

1. Download WLASL and follow the official instructions:  
   WLASL website: <https://dxli94.github.io/WLASL/>

2. Extract 2D body and hand keypoints using a pose estimator such as MMPose (or use your own pose extraction pipeline).

3. Organize data as:

```text
data/
└── wlasl/
    ├── poses/
    │   ├── video_000001.npy
    │   ├── video_000002.npy
    │   └── ...
    ├── wlasl_train_list.txt
    ├── wlasl_val_list.txt
    └── wlasl_test_list.txt
```

4. Optionally, run the preparation script to convert raw pose files into the exact format used by the datasets module:

```bash
python tools/prepare_wlasl_skeleton.py \
    --raw-root data/wlasl/poses \
    --out-root data/wlasl
```

### 4.2 NMFs-CSL

NMFs-CSL is a Chinese Sign Language dataset designed to emphasize non-manual features (facial expressions, mouth shapes, etc.). We follow the official splits and use pre-extracted skeletal sequences when available.

If the dataset is not directly accessible, please refer to `docs/DATASETS.md` for:

- the expected directory structure and file naming convention  
- how to adapt your own CSL data into the same skeleton format

Note: some corpora mentioned in the paper may not be publicly available. For these, we only release the code and configuration templates.

---

## 5. Configuration

All experiment settings are specified via YAML files under `configs/`.

A typical configuration (for example, `configs/wlasl/hma_lbor.yaml`) contains:

- dataset settings (name, root directory, split files, number of classes, number of joints, number of frames)  
- model settings (backbone type, embedding dimension, dropout, depth)  
- LBOR loss settings:  
  - `lambda_lap`: weight of the within-class Laplacian term  
  - `mu_margin`: weight of the center-level margin term  
  - `margin_M`: desired squared Euclidean margin between class centers  
  - `tau`: temperature in the Gaussian kernel for graph edge weights  
  - `use_knn`: whether to sparsify edges using within-class k-nearest neighbours  
  - `knn_k`: number of neighbours when `use_knn` is enabled  
- training settings (epochs, batch size, optimizer, learning rate, weight decay, scheduler, warm-up)  
- data augmentation and normalization settings  
- miscellaneous options (random seed, number of workers, output directory, checkpoint saving frequency)

You can edit these YAML files directly or override specific options from the command line.

---

## 6. Training

After installing dependencies and preparing datasets, you can train LBOR with different backbones using the provided scripts.

Examples:

```bash
# HMA backbone + LBOR on WLASL
bash scripts/train_wlasl_hma_lbor.sh

# SignBERT-style backbone + LBOR on WLASL
bash scripts/train_wlasl_signbert_lbor.sh

# SKIM backbone + LBOR on WLASL
bash scripts/train_wlasl_skim_lbor.sh

# HMA backbone + LBOR on NMFs-CSL
bash scripts/train_nmfscsl_hma_lbor.sh
```

Each script internally calls the training entry:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m lbor_islr.train \
    --config configs/wlasl/hma_lbor.yaml
```

Important command-line arguments (see `src/lbor_islr/train.py`):

- `--config`: path to the YAML configuration file  
- `--resume`: path to a checkpoint to resume training (optional)  
- `--seed`: override the random seed (optional)  
- `--output-dir`: override the output directory (optional)

Training logs (classification loss, Laplacian loss, margin loss, top-1 / top-5 accuracy) and checkpoints are stored under the configured output directory.

---

## 7. Evaluation

To evaluate a trained model on the validation or test set:

```bash
# evaluation of HMA + LBOR on WLASL
bash scripts/eval_wlasl_hma_lbor.sh
```

or equivalently:

```bash
CUDA_VISIBLE_DEVICES=0 python -m lbor_islr.test \
    --config configs/wlasl/hma_lbor.yaml \
    --checkpoint checkpoints/wlasl/hma_lbor_best.pth
```

The evaluation script reports:

- top-1 and top-5 accuracy  
- optionally mean class accuracy and per-class metrics  
- optionally confusion matrices or t-SNE visualizations of the learned embeddings (depending on configuration)

---

## 8. Method Overview

Let `z_i` be the embedding of the i-th training sample and `y_i` its class label.

1. **Within-class Laplacian regularization**

   - For each mini-batch, LBOR builds a within-class graph: edges only connect samples sharing the same label.  
   - Edge weights are defined by a Gaussian kernel on feature distances, optionally restricted to k-nearest neighbours within each class.  
   - A Laplacian (Dirichlet) energy term

     `L_lap = 1/2 * sum_ij A_ij * ||z_i - z_j||^2`

     encourages smooth and connected manifolds within each class, reducing fragmentation into isolated local centroids.

2. **Center-level margin**

   - For each class appearing in the mini-batch, LBOR computes a feature center `mu_c`.  
   - It penalizes pairs of class centers whose squared distance is smaller than a margin `M`, maintaining inter-class separation:

     `L_margin = mean_{c != c'} max(0, M - ||mu_c - mu_c'||^2)`

3. **Total objective**

   - LBOR is added on top of a standard classification loss (e.g. cross-entropy):

     `L_total = L_CE + lambda_lap * L_lap + mu_margin * L_margin`

   - The hyperparameters `lambda_lap`, `mu_margin`, `M` and `tau` (kernel bandwidth) are exposed in the configuration files.

`docs/METHODS.md` provides a more detailed explanation and links each equation to the implementation in `src/lbor_islr/losses/lbor_loss.py`.

---

## 9. Pretrained Models

We provide pretrained checkpoints for selected configurations (e.g. HMA / SignBERT / SKIM backbones with LBOR on WLASL and NMFs-CSL). Because these files can be large, they are not stored directly in this repository.

Please see `checkpoints/README.md` for:

- download links (e.g. GitHub Releases, Google Drive), and  
- instructions on where to place the `.pth` files for evaluation.

---

