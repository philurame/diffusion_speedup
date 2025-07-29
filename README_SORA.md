# Open-Sora-Plan Project Overview

This repository contains the Open-Sora-Plan implementation, covering inference, quantization and timestep optimization methods. Below is a structured breakdown of the project's directories, code modules, and relevant scripts.


## Model Implementations

The core implementations for Open-Sora-Plan models are stored in the `lib/models/sora` directory:

* **Inference Model:** [`r_base_sora.py`](lib/models/sora/r_base_sora.py)
* **Training Model:** [`r_train_sora.py`](lib/models/sora/r_train_sora.py)
* **Quantized Model:** [`r_quant48_sora.py`](lib/models/sora/r_quant48_sora.py)


## Quantization

Quantization-related utilities and functionalities, specifically used for quantizing the Sora model, are located in:

* [`lib/models/models_utils/quantization`](lib/models/models_utils/quantization)

This module is leveraged primarily by the quantized Sora model (`r_quant48_sora.py`).


## Timestep Optimization

Timestep scheduling optimization methods and optimized schedules are stored within:

* [`lib/schedulers`](lib/schedulers)

Optimized timestep schedules specifically used for Open-Sora-Plan are stored in:

* [`r_optimal_sora.py`](lib/schedulers/r_optimal_sora.py)


### Guidance Solvers

Solver implementations used for guidance in diffusion models can be found in the directory:

* [`lib/solvers`](lib/solvers)

The primary solver currently used is the DPM Solver, located at:

* [`r_dpms.py`](lib/solvers/r_dpms.py)


### Training Methods

Training methods for timestep and solver optimization are organized into two main approaches, each stored in its respective subdirectory under [`Training`](Training):

* **Gradient-Based Training:** [`Grad_trainer`](Training/Grad_trainer)
* **Reinforcement Learning (RL) Training:** [`RL_trainer`](Training/RL_trainer)

The core logic for RL-based training is implemented in:

* [`rl.py`](Training/RL_trainer/rl.py)


## Scripts for Experiment Launch

### Configuration Files

Experiment configurations, particularly for RL-based timestep training, are provided in the [`configs`](configs) directory. The primary configuration file used across experiments is:

* [`train_rl.yaml`](configs/train_rl.yaml)


To execute training and other experimental runs, use the scripts located in:

* [`scripts`](scripts)

The primary script for initiating RL-based timestep training is:

* [`train_rl.sh`](scripts/train_rl.sh)



## Setup and Replication Instructions

To successfully replicate experiments or run any Open-Sora-Plan model/method:

1. **Environment Setup:**

   * Create a virtual environment and install dependencies.

2. **Model Preparation:**

   * Load the appropriate Open-Sora-Plan model.

3. **Dataset Preparation:**

   * Prepare your training dataset, optionally including a high-NFE teacher model for guided training.

---
