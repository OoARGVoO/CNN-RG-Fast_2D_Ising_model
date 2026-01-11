# CNN-RG-Fast_2D_Ising_model


### 🤖 AI Disclosure
This project was developed with the assistance of **AI**. The Ai tool helped me with the programming.

The core physical methodology (CNN-based Renormalization Group) and the neural network architecture remain the original research conceptualized by the author.

## Introduction

This repository implements a high-performance **Inverse Renormalization Group (IRG)** framework using Convolutional Neural Networks (CNN). By training a physical 3*3 kernel to capture the statistical characteristics of the 2D Ising model at the critical temperature $T_c$, this framework enables the rapid generation of ultra-large spin configurations (up to $9000^2$ and beyond) while strictly preserving long-range physical correlations.

There are problems with the critical temp here, I used a "Colden" approach, which is clearly wrong, but I couldn't figure out the right way to do that. I'm quite ill these days... If you can help, pls e-mail me.

---

## 🌟 Methodology

In statistical physics, the Renormalization Group (RG) is typically a "coarse-graining" process. This project reverses that flow:
1. **Feature Lock-in**: Training a physical 3*3 kernel to fit multi-scale correlation functions at the critical point ($\beta_c \approx 0.4407$).
2. **Scale Invariance**: Leveraging the fractal nature of the critical state to project small "seed" lattices into larger spatial scales.
3. **Accelerated Reconstruction**: Using a parallelized checkerboard Metropolis algorithm to finalize configurations at high speed, bypassing "Critical Slowing Down." It uses a very low temp to "colden" the spin config, to reduce high-frequency noise. However, this does mean that the spins gets a bit colder.

---

## 📂 File Descriptions

According to the pipeline shown in the repository:

* **`data_collecting.py`**: Generates and saves raw Ising samples at the critical temperature. Outputs `critical_samples.pt`.
* **`kernel_training.py`**: Optimizes the CNN kernel by fitting the multi-scale correlation functions of the training data.
* **`fast_generating_single_stage.py`**: Performs a single $3 \times$ upscaling (e.g., $1000^2 \rightarrow 3000^2$).
* **`fast_generating_multi_stage.py`**: Executes recursive iterations for ultra-high resolution (e.g., $1000^2 \rightarrow 9000^2$ or larger).
* **`normal_fast_2d_Ising.py`**: A standard high-speed Ising simulator used for baseline comparison and seed generation.
* **`rg_model_b3_free_poly.pt` / `.json`**: The trained weights and configuration of the physical model.

---

## 🚀 Quick Start

### 1. Requirements
Install the necessary dependencies:

torch numpy matplotlib

Cuda is also needed. 

### 2. Standard workflow

1. Generating samples: Run `data_collecting.py` to collect Ising configurations at the theoretical critical temperature.

2. Model training: Run `kernel_training.py` to train the 3*3 kernel.

3. Generating new Ising spin configurations: Run `fast_generating_multi_stage.py` or `fast_generating_single_stage.py` . Change the paramiters in these two codes to adjust the size/color/... of the spin config graph.

## VRAM (Video RAM) warning: 
The lattice size grows cubically or quadratically, and 9000x9000 already pushes the limits of many consumer GPUs.

This is a spin config it generates:

![](gallery/ising_iterative_iter2.png)
