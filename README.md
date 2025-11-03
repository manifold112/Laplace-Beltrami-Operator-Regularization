LBOR: Laplace–Beltrami Operator Regularization for Robust Skeleton-based Isolated Sign Language Recognition

This repository provides the official PyTorch implementation of the FG 2026 paper:

LBOR: Laplace–Beltrami Operator Regularization for Robust Skeleton-based Isolated Sign Language Recognition
FG 2026 – IEEE International Conference on Automatic Face and Gesture Recognition

LBOR is a plug-and-play training objective for skeleton-based isolated sign language recognition (ISLR).
It directly regularizes within-class feature geometry by building class-specific subgraphs in the embedding space and minimizing a Laplacian energy, while a lightweight center-level margin preserves between-class separation.
The design is model-agnostic and can be attached to existing ISLR backbones without architectural changes.

🔍 Key Features

Laplace–Beltrami Operator Regularization (LBOR)

Constructs within-class kNN graphs in the feature space for each mini-batch.

Minimizes a Laplacian (Dirichlet) energy to enforce intra-class connectivity and smoothness.

Mitigates signer-driven multi-centroid fragmentation in ISLR.

Center-Level Margin Term

Encourages a margin between class centers in the embedding space.

Preserves inter-class discriminability while LBOR regularizes each class manifold.

Model-Agnostic & Lightweight

No change to the backbone architectures (HMA, SignBERT-style models, SKIM, etc.).

Implemented purely as an additional loss on top of standard classification losses.

Compatible with any skeleton-based encoder that outputs per-instance embeddings.

Reproducible Evaluation on Public ISLR Benchmarks

Word-level American Sign Language (WLASL).

NMFs-CSL (Chinese Sign Language) focusing on non-manual features.

Unified training/evaluation pipeline and configuration files for all reported experiments.

📁 Repository Structure
LBOR-FG2026/
├── README.md
├── LICENSE
├── CITATION.cff
├── FG2026_LBOR.pdf             # (Optional) Paper or preprint
├── requirements.txt            # or environment.yml
├── setup.py / pyproject.toml   # (Optional) install as a package
├── .gitignore

├── configs/                    # Experiment configurations
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
│       │   ├── lbor_loss.py    # LBOR implementation (Laplacian + center margin)
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
│       ├── train.py            # Training entry point
│       └── test.py             # Evaluation entry point

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
│   └── README.md               # Links to pretrained weights (not stored in git)

└── figures/
    ├── method_overview.png
    ├── laplacian_graph.png
    └── center_margin.png


You do not need to strictly follow this layout, but a similar separation between configs, core code, scripts, tools, docs, checkpoints, and figures is recommended for clarity and reproducibility.

📦 Installation

We recommend using a Conda environment:

# 1. Create environment
conda create -n lbor_islr python=3.10 -y
conda activate lbor_islr

# 2. Clone this repository
git clone <repository-url>
cd LBOR-FG2026

# 3. Install dependencies
pip install -r requirements.txt

# (Optional) Install as a package
pip install -e .


The implementation has been tested with:

Python ≥ 3.9

PyTorch ≥ 1.12 (with a matching CUDA toolkit)

torchvision, PyYAML, NumPy, SciPy, tqdm, and other standard libraries

More detailed notes on environment setup and compatible versions are provided in docs/INSTALL.md.

📚 Datasets
WLASL

We evaluate LBOR on WLASL (Word-Level American Sign Language) using 2D skeleton sequences extracted from RGB videos.

Download WLASL and follow the official instructions:

WLASL website: https://dxli94.github.io/WLASL/

Extract 2D body and hand keypoints using a pose estimator such as MMPose:

# Example (pseudo-code):
python tools/extract_wlasl_poses_with_mmpose.py \
    --wlasl-root /path/to/WLASL \
    --out-root data/wlasl/poses


Organize data as:

data/
└── wlasl/
    ├── poses/
    │   ├── video_000001.npy
    │   ├── video_000002.npy
    │   └── ...
    ├── wlasl_train_list.txt
    ├── wlasl_val_list.txt
    └── wlasl_test_list.txt


Optionally, run the preparation script to convert raw pose files into the exact format used by the datasets module:

python tools/prepare_wlasl_skeleton.py \
    --raw-root data/wlasl/poses \
    --out-root data/wlasl

NMFs-CSL

NMFs-CSL is a Chinese Sign Language dataset designed to emphasize non-manual features (facial expressions, mouth shapes, etc.).
We follow the official split and use pre-extracted skeletal sequences when available.

If the dataset is not directly accessible, please refer to docs/DATASETS.md for:

The expected directory structure and file naming convention.

How to adapt your own CSL data into the same skeleton format.

Note: Some corpora mentioned in the paper (e.g., SLR500, MS-ASL) are not used in our experiments due to limited or unstable public availability, which prevents fair and fully reproducible comparison.

⚙️ Configuration

All experiment settings are specified via YAML configuration files under configs/.

A typical configuration (e.g., configs/wlasl/hma_lbor.yaml) contains:

Dataset

Dataset name, root directory, split files.

Number of classes, number of joints, number of frames (we resample each clip to a fixed number of frames).

Model

Backbone type: hma, signbert, skim.

Embedding dimension (feature dimension).

Dropout rate, layer counts, and other architecture hyperparameters.

Loss (LBOR)

lambda_lap: weight of the within-class Laplacian term.

mu_margin: weight of the center-level margin term.

margin_M: desired squared Euclidean margin between class centers.

tau: temperature in the Gaussian kernel for graph edge weights.

use_knn: whether to sparsify edges using within-class kNN.

knn_k: number of neighbors when use_knn is enabled.

Training

Number of epochs, batch size, optimizer type (e.g., AdamW), base learning rate, weight decay.

Learning rate scheduler (e.g., cosine decay) and warmup epochs.

Augmentation

Temporal resampling strategy, random cropping, random flipping/scaling.

Skeleton normalization (translation, scaling) and joint selection.

Misc

Random seed, number of dataloader workers.

Output directory for logs and checkpoints.

Checkpoint saving frequency.

You can edit these YAML files directly or override specific options from the command line.

🚀 Training

After installing dependencies and preparing datasets, training LBOR on WLASL with a chosen backbone can be done using the provided scripts:

# Example: HMA backbone + LBOR on WLASL
bash scripts/train_wlasl_hma_lbor.sh

# Example: SignBERT-style backbone + LBOR on WLASL
bash scripts/train_wlasl_signbert_lbor.sh

# Example: SKIM backbone + LBOR on WLASL
bash scripts/train_wlasl_skim_lbor.sh

# Example: HMA backbone + LBOR on NMFs-CSL
bash scripts/train_nmfscsl_hma_lbor.sh


Each script internally calls the training entry point:

CUDA_VISIBLE_DEVICES=0,1 python -m lbor_islr.train \
    --config configs/wlasl/hma_lbor.yaml


Main command-line arguments (see src/lbor_islr/train.py):

--config: path to the YAML config file.

--resume: path to a checkpoint to resume training (optional).

--seed: optional override of the random seed.

--output-dir: optional override of the output directory.

Training logs (including classification loss, Laplacian loss, margin loss, Top-1/Top-5 accuracy) will be stored under the configured output directory.

✅ Evaluation

To evaluate a trained model on the validation or test set:

# Example: evaluation of HMA + LBOR on WLASL
bash scripts/eval_wlasl_hma_lbor.sh


or directly:

CUDA_VISIBLE_DEVICES=0 python -m lbor_islr.test \
    --config configs/wlasl/hma_lbor.yaml \
    --checkpoint checkpoints/wlasl/hma_lbor_best.pth


The evaluation script reports:

Top-1 and Top-5 accuracy.

Optionally, mean class accuracy and per-class metrics.

Optionally, confusion matrices and t-SNE visualizations of the learned embeddings (config-dependent).

