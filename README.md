# LBOR: Laplace–Beltrami Operator Regularization for Robust Skeleton-based Isolated Sign Language Recognition

This repository provides the **official PyTorch implementation** of the FG 2026 paper:

> **LBOR: Laplace–Beltrami Operator Regularization for Robust Skeleton-based Isolated Sign Language Recognition**  
> FG 2026 – IEEE International Conference on Automatic Face and Gesture Recognition

LBOR is a **plug-and-play training objective** for skeleton-based isolated sign language recognition (ISLR).  
It directly regularizes **within-class feature geometry** by building class-specific subgraphs in the embedding space and minimizing a Laplacian energy, while a lightweight **center-level margin** preserves **between-class separation**.  
The design is **model-agnostic** and can be attached to existing ISLR backbones without architectural changes.

---

## 🔍 Key Features

- **Laplace–Beltrami Operator Regularization (LBOR)**  
  - Constructs **within-class kNN graphs** in the feature space for each mini-batch.  
  - Minimizes a Laplacian (Dirichlet) energy to enforce **intra-class connectivity and smoothness**.  
  - Mitigates signer-driven **multi-centroid fragmentation** in ISLR.

- **Center-Level Margin Term**  
  - Encourages a margin between class centers in the embedding space.  
  - Preserves **inter-class discriminability** while LBOR regularizes each class manifold.

- **Model-Agnostic & Lightweight**  
  - No change to the backbone architectures (HMA, SignBERT-style models, SKIM, etc.).  
  - Implemented purely as an additional loss on top of standard classification losses.  
  - Compatible with any skeleton-based encoder that outputs per-instance embeddings.

- **Reproducible Evaluation on Public ISLR Benchmarks**  
  - Word-level American Sign Language (WLASL).  
  - NMFs-CSL (Chinese Sign Language) focusing on non-manual features.  
  - Unified training/evaluation pipeline and configuration files for all reported experiments.

---

## 📁 Repository Structure

```text
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
