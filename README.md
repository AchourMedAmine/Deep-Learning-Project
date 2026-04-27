# 🫁 Respiratory Sound Classification with SAM-Optimized Audio Spectrogram Transformers

[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
[![Dataset](https://img.shields.io/badge/Dataset-ICBHI%202017-green?style=for-the-badge)](https://bhichallenge.med.auth.gr/)

> **Paper**: *"Geometry-Aware Optimization for Respiratory Sound Classification: Enhancing Sensitivity with SAM-Optimized Audio Spectrogram Transformers"*
>
> **Reference repo**: [Atakanisik/ICBHI-AST-SAM](https://github.com/Atakanisik/ICBHI-AST-SAM)

This project **reproduces** the paper's baseline results and then **proposes amelioration experiments** targeting higher sensitivity for reliable clinical screening of respiratory pathologies (Crackles, Wheezes, Both) on the ICBHI 2017 Challenge dataset.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [End-to-End Pipeline](#-end-to-end-pipeline)
- [Project Architecture](#-project-architecture)
- [Model Architecture](#-model-architecture)
- [Training Strategy](#-training-strategy)
- [Evaluation Protocol](#-evaluation-protocol)
- [Getting Started](#-getting-started)
- [Amelioration Experiments](#-amelioration-experiments)
- [Target Metrics](#-target-metrics)

---

## 🎯 Project Overview

Respiratory diseases like COPD and asthma produce abnormal lung sounds (crackles and wheezes) that clinicians detect via auscultation. Automated classification of these sounds can assist in early screening. This project tackles three key challenges:

| Challenge | Our Approach |
|:---|:---|
| **Small medical dataset** (920 recordings, 6898 cycles) | Transfer learning from AudioSet-pretrained AST |
| **Severe class imbalance** (Normal ≫ Abnormal subtypes) | Weighted Random Sampler + Label Smoothing |
| **Overfitting to sharp minima** | Sharpness-Aware Minimization (SAM) optimizer |
| **Short/variable-length signals** | Signal-preserving Cyclic Padding (no zero-padding) |

---

## 🔄 End-to-End Pipeline

```
┌──────────────┐    ┌────────────────┐    ┌───────────────┐    ┌──────────────┐    ┌────────────────┐
│  Raw Audio   │───▶│ Preprocessing  │───▶│   AST Model   │───▶│   Training   │───▶│  Evaluation    │
│  (.wav)      │    │                │    │                │    │              │    │                │
│  920 files   │    │ • Resample 16k │    │ • ViT Encoder  │    │ • SAM optim  │    │ • Binary eval  │
│  + .txt ann  │    │ • Segment      │    │ • 12 layers    │    │ • WRS sampler│    │ • Se, Sp, Score│
│              │    │ • Cyclic Pad   │    │ • Mean Pool    │    │ • 20 epochs  │    │ • Conf Matrix  │
│              │    │ • Log-Mel Spec │    │ • 4-class head │    │ • BS=8       │    │                │
└──────────────┘    └────────────────┘    └───────────────┘    └──────────────┘    └────────────────┘
```

### Preprocessing Detail

```
Raw Audio ──▶ Resample to 16kHz ──▶ Segment by annotation ──▶ Cyclic Padding to 8s
                                                                     │
                                           ┌─────────────────────────┘
                                           ▼
                              Signal < 8s? Repeat cyclically
                              Signal ≥ 8s? Truncate to 8s
                                           │
                                           ▼
                              ASTFeatureExtractor ──▶ Log-Mel Spectrogram ──▶ 16×16 Patches
```

> **Why Cyclic Padding?** Zero-padding introduces artificial silence that can mislead the classifier. Cyclic repetition preserves the pathological patterns (crackles/wheezes) and maximizes the signal available for each respiratory cycle.

---

## 🏗️ Project Architecture

Config-driven design — every experiment variant is defined in a YAML file, no code changes needed.

```
Deep-Learning-Project/
│
├── 📂 config/                       # ⚙️ Experiment Configurations
│   ├── baseline.yaml                #   Reproduce the paper (Phase 5)
│   ├── exp_p1_wavelets.yaml         #   Wavelet-based preprocessing
│   ├── exp_p2_focal.yaml            #   Focal Loss for imbalance
│   ├── exp_p3_asam.yaml             #   Adaptive SAM optimizer
│   └── exp_p4_ssast.yaml            #   SSAST architecture swap
│
├── 📂 src/                          # 🧠 Core Source Code
│   │
│   ├── 📂 data/                     #   Data Pipeline
│   │   ├── preprocess.py            #     Raw .wav → segmented, padded .npz
│   │   ├── dataset.py               #     PyTorch Dataset + ASTFeatureExtractor
│   │   └── augment.py               #     Gain perturbation, Gaussian noise, SpecAugment
│   │
│   ├── 📂 models/                   #   Model Architectures
│   │   ├── base_ast.py              #     ✅ AST baseline (paper reproduction)
│   │   ├── ssast.py                 #     🔬 Self-Supervised AST (amelioration)
│   │   ├── htsat.py                 #     🔬 Hierarchical Token-Semantic AT
│   │   └── lungadapter.py           #     🔬 Adapter-based fine-tuning
│   │
│   ├── 📂 optim/                    #   Optimizers & Loss Functions
│   │   ├── sam.py                   #     ✅ SAM optimizer (paper reproduction)
│   │   ├── asam.py                  #     🔬 Adaptive SAM (amelioration)
│   │   ├── esam.py                  #     🔬 Efficient SAM
│   │   └── losses.py                #     CE (baseline) + Focal Loss (amelioration)
│   │
│   ├── 📂 utils/                    #   Shared Utilities
│   │   ├── seed.py                  #     Fix all random seeds for reproducibility
│   │   ├── metrics.py               #     ICBHI binary Se/Sp/Score computation
│   │   └── logger.py                #     CSV + TensorBoard experiment logging
│   │
│   ├── train.py                     #   🚀 Config-driven training loop
│   └── evaluate.py                  #   🔍 Model evaluation & visualization
│
├── 📂 data/
│   ├── raw/                         #   Original ICBHI audio + annotations
│   └── processed/                   #   Preprocessed .npz files
│
├── 📂 notebooks/                    #   📓 Exploration & Prototyping
│   ├── P1_exploration/              #     Preprocessing experiments
│   ├── P2_exploration/              #     Loss function experiments
│   ├── P3_exploration/              #     Optimizer experiments
│   └── P4_exploration/              #     Architecture experiments
│
├── 📂 results/                      #   📈 Saved models, metrics, plots
├── requirements.txt                 #   📦 Python dependencies
└── README.md
```

---

## 🧠 Model Architecture

### Audio Spectrogram Transformer (AST) — Baseline

The core model is a Vision Transformer (ViT) adapted for audio. It was pretrained on **AudioSet** (~2M audio clips) and fine-tuned on the ICBHI dataset.

```
                        Input: Raw waveform (16kHz × 8s = 128,000 samples)
                                          │
                                          ▼
                         ┌────────────────────────────────┐
                         │    ASTFeatureExtractor          │
                         │    (Log-Mel Spectrogram)        │
                         │    128 mel bins × 1024 frames   │
                         └───────────────┬────────────────┘
                                         │
                                         ▼
                         ┌────────────────────────────────┐
                         │    Patch Embedding (16×16)      │
                         │    + Positional Encoding        │
                         │    → Sequence of patch tokens   │
                         └───────────────┬────────────────┘
                                         │
                                         ▼
                         ┌────────────────────────────────┐
                         │    12× Transformer Encoder      │
                         │    ┌────────────────────────┐  │
                         │    │ Multi-Head Attention    │  │
                         │    │ (12 heads, 768-dim)     │  │
                         │    ├────────────────────────┤  │
                         │    │ Feed-Forward Network    │  │
                         │    │ (768 → 3072 → 768)     │  │
                         │    └────────────────────────┘  │
                         └───────────────┬────────────────┘
                                         │
                                         ▼
                         ┌────────────────────────────────┐
                         │    Mean Pooling                 │
                         │    (average over all patches)   │
                         │    → 768-dim embedding          │
                         └───────────────┬────────────────┘
                                         │
                                         ▼
                         ┌────────────────────────────────┐
                         │    Classification Head          │
                         │    Dropout(0.3) → Linear(768→4)│
                         │    → [Normal, Crackle,          │
                         │       Wheeze, Both]             │
                         └────────────────────────────────┘
```

> **Key design choice**: We use **Mean Pooling** over all patch embeddings instead of the CLS token. This provides more stable gradient flow for the small ICBHI dataset.

---

## ⚡ Training Strategy

### SAM Optimizer (Sharpness-Aware Minimization)

SAM seeks parameters that lie in **flat minima** of the loss landscape, which generalize better than sharp minima. Each training step requires **two forward-backward passes**:

```
Standard SGD:        θ ← θ − η∇L(θ)              (1 pass)

SAM:                 ε  = ρ · ∇L(θ) / ‖∇L(θ)‖    (perturbation)
                     θ ← θ − η∇L(θ + ε)           (2 passes)
```

### Training Configuration

| Hyperparameter | Value | Rationale |
|:---|:---|:---|
| Base Optimizer | AdamW | Decoupled weight decay for transformers |
| Learning Rate | 1×10⁻⁵ | Small LR for fine-tuning pretrained model |
| Weight Decay | 1×10⁻⁴ | Regularization |
| SAM ρ | 0.05 | Neighborhood radius for perturbation |
| Batch Size | 8 | Limited by GPU memory (Tesla L4) |
| Epochs | 20 | Sufficient for convergence |
| Loss | CrossEntropy | With label smoothing = 0.1 |
| Sampler | WeightedRandomSampler | Balances class distribution during training |
| Augmentation | Gain ± 10%, Noise σ=0.0001 | Light augmentation, 50% probability each |

### Class Imbalance Handling

The ICBHI dataset is heavily imbalanced:

```
Normal  ████████████████████████████████████████  ~3642 (53%)
Crackle ██████████████████                        ~1864 (27%)
Wheeze  ████████                                   ~886 (13%)
Both    ███                                         ~506 ( 7%)
```

**WeightedRandomSampler** assigns each sample a weight inversely proportional to its class frequency, ensuring the model sees an equal distribution of all classes during training.

---

## 📏 Evaluation Protocol

Following the **ICBHI 2017 Official Protocol** — 4-class predictions are collapsed into binary evaluation:

```
             Predicted
             ┌──────────┬──────────┐
             │  Normal  │ Abnormal │
    ┌────────┼──────────┼──────────┤
    │ Normal │   TN     │    FP    │   ← Specificity = TN / (TN + FP)
True├────────┼──────────┼──────────┤
    │Abnormal│   FN     │    TP    │   ← Sensitivity = TP / (TP + FN)
    └────────┴──────────┴──────────┘

    where Abnormal = {Crackle, Wheeze, Both}

    ICBHI Score = (Sensitivity + Specificity) / 2
```

A prediction counts as **True Positive** if any abnormal class is predicted for a truly abnormal sample (even if the specific abnormal subtype is wrong).

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Clone this repository
git clone <your-repo-url>
cd Deep-Learning-Project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

**Hardware**: NVIDIA Tesla L4 GPU (Google Colab) or equivalent. PyTorch with CUDA support required.

### 2. Dataset Acquisition

Download the ICBHI 2017 Challenge dataset (920 recordings, 6898 respiratory cycles):

1. Download [ICBHI_final_database.zip](https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip)
2. Extract into `data/raw/ICBHI_final_database/`
3. Download [ICBHI_challenge_train_test.txt](https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_challenge_train_test.txt) into `data/raw/`

Expected structure:
```
data/raw/
├── ICBHI_final_database/
│   ├── 101_1b1_Al_sc_Meditron.wav
│   ├── 101_1b1_Al_sc_Meditron.txt
│   └── ... (920 recordings)
└── ICBHI_challenge_train_test.txt
```

### 3. Preprocessing

```bash
python src/data/preprocess.py
```
Outputs `data/processed/icbhi_ast_16k_8s.npz` containing cyclic-padded waveforms split by the official 60/40 partition.

### 4. Training (Baseline Reproduction)

```bash
python src/train.py --config config/baseline.yaml
```

### 5. Evaluation

```bash
python src/evaluate.py --config config/baseline.yaml
```
Generates confusion matrix and computes Se / Sp / Score.

---

## 🧪 Amelioration Experiments (Phase 6)

After reproducing the baseline, we explore four improvement axes:

| Phase | Axis | Hypothesis | Config | Key Change |
|:---:|:---|:---|:---|:---|
| **P1** | 🔊 Preprocessing | Wavelet transforms capture multi-resolution time-frequency info better than Mel spectrograms | `exp_p1_wavelets.yaml` | Replace Mel → CWT |
| **P2** | ⚖️ Loss Function | Focal Loss focuses learning on hard, minority-class examples | `exp_p2_focal.yaml` | Replace CE → Focal Loss |
| **P3** | 📐 Optimizer | Adaptive SAM scales perturbation per-parameter for heterogeneous transformer weights | `exp_p3_asam.yaml` | Replace SAM → ASAM |
| **P4** | 🏛️ Architecture | Self-supervised pretraining (SSAST) learns better audio representations from unlabeled data | `exp_p4_ssast.yaml` | Replace AST → SSAST |

Run any experiment:
```bash
python src/train.py --config config/exp_p2_focal.yaml
```

---

## 📊 Target Metrics

### Baseline Reproduction (Paper Results)

| Metric | Target | Description |
|:---|:---:|:---|
| **Sensitivity (Se)** | **68.31%** | Abnormal correctly detected |
| **Specificity (Sp)** | **67.89%** | Normal correctly identified |
| **ICBHI Score** | **68.10%** | (Se + Sp) / 2 |

### Amelioration Goal
> Improve **Sensitivity (recall)** beyond 68.31% while maintaining competitive Specificity.

---

## 📚 References

- **Paper**: Isik et al., *"Geometry-Aware Optimization for Respiratory Sound Classification"*
- **AST**: Gong et al., [*"AST: Audio Spectrogram Transformer"*](https://arxiv.org/abs/2104.01778), INTERSPEECH 2021
- **SAM**: Foret et al., [*"Sharpness-Aware Minimization for Efficiently Improving Generalization"*](https://arxiv.org/abs/2010.01412), ICLR 2021
- **ICBHI 2017**: Rocha et al., *"An open access database for the evaluation of respiratory sound classification algorithms"*, Physiological Measurement, 2019

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
