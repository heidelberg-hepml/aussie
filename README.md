# AUSSIE — Adversary-free Unfolding SanS Iteration or Emulation

AUSSIE is a non-iterative, discriminative method for unbinned unfolding of multidimensional collider observables. It reweights a simulated reference dataset to correct for detector effects without requiring iterations, adversarial training, or surrogate forward models.

> **Paper** &nbsp; *"Unfolding without Iterations, Adversaries, or Surrogates"*
> Ayodele Ore and Tilman Plehn — [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

## Overview

Unfolding is the inverse problem of deconvolving observed (reco-level) measurements to recover the underlying truth (part-level) distributions. AUSSIE phrases this problem in terms of density ratios and solves it in two steps:

1. **Classification** — Train a classifier $R_\theta(x)$ to estimate the reco-level density ratio $p_\text{data}(x) / p_\text{sim}(x)$.
2. **Unfolding** — Train a part-level network $\overline R_\varphi(z)$ such that its forward mapping matches $R_\theta(x)$, by minimizing the RKHS norm of the functional gradient of an MLC loss. This directly yields part-level weights that solve the unfolding integral equation — no iterations needed.

Two loss functions are provided for the second step:

| Loss | Best for | Description |
|------|----------|-------------|
| **Gaussian kernel** | Low-dimensional observables | Analytic kernel with a scale hyperparameter $\Lambda$ |
| **AutoDiff** | High-dimensional / point-cloud data | Uses the Neural Tangent Kernel via `autograd`; no extra hyperparameters |


## Project Structure

```
aussie/
├── aussie.py               # Entry point (Hydra)
├── config/                 # Hydra configuration files
│   ├── default.yaml        #   Global defaults
│   ├── cls.yaml            #   Classification experiment
│   ├── unf.yaml            #   Unfolding experiment
│   ├── itr.yaml            #   OmniFold iteration experiment
│   ├── dataset/            #   Dataset configs (zjet, gaussian_toy, yukawa, …)
│   ├── net/                #   Network configs (mlp, lgatr, …)
│   └── cluster/            #   Cluster submission configs
├── src/
│   ├── datasets/           # Dataset readers & preprocessors
│   │   ├── gaussian.py     #   Gaussian toy example
│   │   ├── zjet.py         #   Zj substructure observables
│   │   ├── zjet_particle.py#   Zj jet constituents
│   │   └── yukawa.py       #   tHj parton-level events
│   ├── experiments/        # Experiment logic
│   │   ├── classification.py   # Step 1: reco-level classifier
│   │   ├── unfolding.py        # Step 2: AUSSIE unfolder
│   │   └── iteration.py        # OmniFold iterative baseline
│   ├── models/             # Model wrappers
│   │   ├── classifier.py   #   Classifier model
│   │   └── unfolder.py     #   Unfolder model (Gauss & AutoDiff losses)
│   ├── networks/           # Neural network architectures
│   │   ├── mlp.py          #   Multi-layer perceptron
│   │   ├── lgatr.py        #   Lorentz Geometric Algebra Transformer
│   │   └── transformer.py  #   Transformer backbone
│   └── utils/              # Utilities
│       ├── trainer.py      #   Training loop
│       ├── plotting.py     #   Result plotting
│       ├── transforms.py   #   Data preprocessing
│       └── cluster.py      #   SLURM job submission
├── env.yaml                # Conda environment specification
├── setup.sh                # Environment setup script (for clusters)
└── LICENSE                 # MIT License
```

## Installation

Create the conda environment and install dependencies:

```bash
conda env create -f env.yaml
conda activate aussie
```

**Requirements:** Python 3.10, PyTorch, Hydra, NumPy, Matplotlib, Pandas, scikit-learn, SciPy, TensorBoard, [schedulefree](https://github.com/facebookresearch/schedule_free), [energyflow](https://energyflow.network/) (for the Zj dataset).

For constituent-level experiments using LGATr, `xformers` and `jvp_flash_attention` are additionally required.

## Usage

AUSSIE is configured and launched with [Hydra](https://hydra.cc). All experiments are run through the same entry point:

```bash
python aussie.py --config-name <CONFIG> [overrides]
```

### Step 1 — Classification

Train the reco-level density ratio classifier:

```bash
python aussie.py --config-name cls/gaussian
```

### Step 2 — Unfolding (AUSSIE)

Train the part-level unfolder, pointing to the trained classifier:

```bash
python aussie.py --config-name unf/gaussian cls_path=<path/to/classifier/run>
```

### OmniFold Baseline

Run iterative OmniFold for comparison:

```bash
python aussie.py --config-name itr/gaussian iterations=10
```

### Common Overrides

Override any config value from the command line:

```bash
# Change batch size and learning rate
python aussie.py --config-name unf ... training.batch_size=2048 training.lr=5e-4

# Use a different dataset
python aussie.py --config-name cls/yukawa

# Chain AUSSIE steps 1 & 2 with the iteration experiment
python aussie.py --config-name itr/yukawa model._target_=src.models.Unfolder

# Skip training and only evaluate / plot existing results
python aussie.py prev_exp_dir=<path/to/run> train=false
```

### Available Datasets

| Config name | Description |
|---|---|
| `gaussian` | 1D Gaussian toy example |
| `zjet` | Jet substructure observables (6D) |
| `zjet_corrected` | Jet substructure observables with hidden variables correction (6D) |
| `zjet_particle` | Jet constituents (full phase space) |
| `yukawa` | $pp \to tHj$ parton-level events |

## Outputs

Each run is saved to `runs/<run_name>/<timestamp>/` and contains:

- **Hydra config** — full resolved configuration (`.hydra/`)
- **Checkpoints** — best and optionally periodic model checkpoints
- **TensorBoard logs** — training / validation loss curves
- **Plots** — reco- and part-level distribution comparisons

View training curves with:

```bash
tensorboard --logdir runs/
```

## Citation

If you use this code, please cite:

```bibtex
@article{Ore:2025aussie,
    author  = {Ore, Ayodele and Plehn, Tilman},
    title   = {Unfolding without Iterations, Adversaries, or Surrogates},
    year    = {2026},
    eprint  = {XXXX.XXXXX},
    archivePrefix = {arXiv},
    primaryClass  = {hep-ph}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
