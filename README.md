# UIESNN: Scale-Aware Spiking Network for Underwater Image Enhancement (IJCNN 2026)

This repository contains the official implementation of **UIESNN**, a scale-aware Spiking Neural Network (SNN) for underwater image enhancement.

Underwater images often suffer from color casts, haze-like blur, and low contrast due to light attenuation and scattering. Unlike common image restoration tasks that focus on high-frequency noise or rain streaks, underwater image enhancement requires modeling large-scale and low-frequency degradations. UIESNN addresses this problem with a spike-driven architecture designed for efficient and coherent underwater image restoration.

## Highlights

- **Scale-aware spiking design** for underwater image enhancement
- **Multi-scale Pooling LIF Block (MPLB)** to enlarge the receptive field of spiking neurons
- **Spiking Residual Block (SRB)** combining frequency decomposition, MPLB, and attention refinement
- Fully spike-driven encoder-decoder framework
- Competitive restoration quality with low energy cost

## Method Overview

UIESNN is built around the **Multi-scale Pooling LIF Block (MPLB)**. Instead of relying only on local spiking responses, MPLB uses multi-scale average pooling to inject local, regional, and global context into the membrane dynamics of spiking neurons.

This helps the network capture:

- global color casts
- haze-like low-frequency degradation
- regional illumination changes
- fine texture and edge details

The full network adopts an encoder-decoder structure with stacked **Spiking Residual Blocks**, multi-scale feature injection, and multi-scale supervision.

## Architecture

The main components are:

1. **Embedding Layer**  
   Converts the input underwater image into feature space.

2. **Spiking Residual Block (SRB)**  
   Performs feature refinement using:
   - Frequency Decomposition Module
   - Multi-scale Pooling LIF Block
   - Multi-dimensional Attention

3. **Encoder-Decoder Network**  
   Extracts and reconstructs multi-scale features using spike-driven operations.

4. **Multi-scale Reconstruction Heads**  
   Produces enhanced outputs at multiple resolutions during training.

## Results

UIESNN is evaluated on the **EUVP** and **LSUI** underwater image enhancement benchmarks.

| Dataset | PSNR | SSIM |
|---|---:|---:|
| EUVP | 26.97 | 0.8936 |
| LSUI | 24.7346 | 0.8754 |

Compared with previous SNN-based restoration methods, UIESNN achieves better color fidelity, improved spatial coherence, and competitive energy efficiency.

## Installation

```bash
git clone https://github.com/ChrisChen1023/UIESNN.git
cd UIESNN

conda create -n uiesnn python=3.9
conda activate uiesnn

pip install -r requirements.txt
