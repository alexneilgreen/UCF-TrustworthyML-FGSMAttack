<!-- SHOWCASE: true -->

# FGSM White-Box Adversarial Attack

> Implements and evaluates Fast Gradient Sign Method (FGSM) attacks under an L-infinity threat model on ResNet-18 and Vision Transformer architectures across MNIST and CIFAR-10 datasets.

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Language](https://img.shields.io/badge/language-Python-blue)
![Semester](https://img.shields.io/badge/semester-Fall%202025-orange)

---

## Course Information

| Field                  | Details                                                                                                                                                                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Course Title           | Trustworthy Machine Learning                                                                                                                                                                                                                                  |
| Course Number          | CAP6938                                                                                                                                                                                                                                                       |
| Semester               | Fall 2025                                                                                                                                                                                                                                                     |
| Assignment Title       | White-Box Attack (FGSM)                                                                                                                                                                                                                                       |
| Assignment Description | Implement and evaluate FGSM attacks under an L-infinity threat model on two architectures (ResNet-18, ViT) and two datasets (MNIST, CIFAR-10). Compare targeted vs. untargeted objectives and report robustness metrics with clear, reproducible experiments. |

---

## Project Description

This project implements the Fast Gradient Sign Method (FGSM) from scratch to evaluate adversarial robustness of two neural network architectures. Both untargeted and targeted attack variants are supported, with targeted attacks using either a random class or the least-likely class derived from clean model logits. Experiments sweep perturbation budgets epsilon in {1, 2, 4, 8}/255 on a fixed 1,000-image test subset per dataset, producing robustness tables, attack success rates, and perturbation visualizations. Key findings show that ResNet-18 is nearly fully compromised on MNIST at even epsilon = 1/255, while ViT retains moderate robustness due to its self-attention smoothing effect.

---

## Screenshots / Demo

> _No screenshot available. Add one with: `![Demo](docs/your-image.png)`_

---

## Results

Running the full pipeline produces terminal output and saves files to `./results/`. A sample of expected terminal output:

```
Using device: cuda

Clean Accuracy: 99.89%

Evaluating epsilon = 0.00392
  Robust Acc: 9.70%
  Untargeted ASR: 90.30%
  Targeted ASR (Random): 10.20%
  Targeted ASR (Least-likely): 0.00%
```

**Key output files:**

```
results/
├── !resnet18_mnist_training.json       # ResNet-18 MNIST training parameters
├── !resnet18_cifar10_training.json     # ResNet-18 CIFAR-10 training parameters
├── !vit_mnist_training.json            # ViT MNIST training parameters
├── !vit_cifar10_training.json          # ViT CIFAR-10 training parameters
├── 0_clean_accuracy_results.json       # Clean accuracy for all models
├── 0_fgsm_task2_results.json           # Full robustness evaluation results
├── 1_clean_accuracy.png                # Bar chart: clean accuracy comparison
├── 2_clean_training.png                # Training curves for all models
├── 3_fgsm_visualization_*.png          # Attack visualizations at epsilon = 8/255
├── 4_table_mnist_results.csv           # MNIST results table
└── 4_table_cifar10_results.csv         # CIFAR-10 results table
```

**Interpreting the results:** robust accuracy should decrease and ASR should increase as epsilon grows. If robust accuracy is unexpectedly high, verify that the test subset is being created with the seeded indices and that input normalization is consistent between training and attack evaluation.

**Achieved results:**

| Model     | Dataset  | Clean Acc | Robust Acc (8/255) | Untargeted ASR (8/255) |
| --------- | -------- | --------- | ------------------ | ---------------------- |
| ResNet-18 | MNIST    | 99.89%    | 9.7%               | 90.3%                  |
| ResNet-18 | CIFAR-10 | 91.33%    | 31.7%              | 68.3%                  |
| ViT       | MNIST    | 98.40%    | 32.9%              | 67.1%                  |
| ViT       | CIFAR-10 | 82.54%    | 28.6%              | 71.4%                  |

---

## Key Concepts

`adversarial-examples` `fgsm` `l-infinity-threat-model` `untargeted-attack` `targeted-attack` `least-likely-class` `adversarial-robustness` `attack-success-rate` `vision-transformer` `resnet`

---

## Languages & Tools

- **Language:** Python 3.7+
- **Framework:** PyTorch, torchvision
- **Build System:** pip / requirements.txt

---

## File Structure

```
.
├── main.py        # Entry point - handles training, evaluation, and visualization
├── resnet.py      # ResNet-18 implementation adapted for 28x28 and 32x32 inputs
├── vit.py         # Vision Transformer implementation for small image sizes
├── fgsm.py        # FGSM attack functions (untargeted + targeted) and robustness evaluation
├── README.md      # This file
├── data/          # Dataset downloads (auto-created by torchvision)
├── models/        # Saved model weights (auto-created)
└── results/       # All output files: JSON results, plots, CSVs (auto-created)
```

---

## Installation & Usage

### Prerequisites

- Python 3.7+
- GPU strongly recommended for training (CUDA-capable device)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/alexneilgreen/UCF-TrustworthyML-FGSMAttack.git
cd UCF-TrustworthyML-FGSMAttack

# 2. Install dependencies
pip install -r requirements.txt
```

### Running

```bash
# Run the full pipeline (train models, then evaluate attacks)
python main.py

# Task 1 only: train models and report clean accuracy
python main.py --task 1

# Task 2 only: run full FGSM evaluation (requires Task 1 models to exist)
python main.py --task 2
```

### CLI Options

| Argument   | Values  | Action                                        |
| ---------- | ------- | --------------------------------------------- |
| `--task 0` | Default | Run both Task 1 and Task 2                    |
| `--task 1` | -       | Train models and evaluate clean accuracy only |
| `--task 2` | -       | Run full FGSM robustness evaluation only      |

---

## Model Configurations

**ResNet-18:**

- Adapted for 28x28 (MNIST, 1 input channel) and 32x32 (CIFAR-10, 3 input channels)
- Optimizer: Adam (lr=0.001, weight decay=1e-4)
- Scheduler: StepLR (step=5, gamma=0.1)
- Epochs: 10 (MNIST), 30 (CIFAR-10)

**Vision Transformer:**

- Patch size: 4x4, embed dim: 256, depth: 8 layers, 4 attention heads
- Optimizer: AdamW (lr=0.00025, weight decay=0.05)
- Scheduler: CosineAnnealingLR
- Epochs: 10 (MNIST), 30 (CIFAR-10) with early stopping (patience=5) and gradient clipping (max norm=1.0)

---

## Reproducibility

All experiments use fixed seeds to ensure reproducibility:

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

The 1,000-image test subset is drawn using the seeded shuffled indices, ensuring the same images are evaluated across runs.

---

## Academic Integrity

This repository is publicly available for **portfolio and reference purposes only**.
Please do not submit any part of this work as your own for academic coursework.
