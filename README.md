# Transformer Implementation (Attention Is All You Need)

This repository contains a faithful PyTorch implementation of the Transformer model from the landmark paper ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762).

In addition to the standard architecture, this project implements modern optimizations (such as **Pre-Layer Normalization**) to improve training stability, with a roadmap for further state-of-the-art enhancements.

## 🚀 Current Status & Features

### 1\. Baseline Architecture (Implemented)

[cite_start]We have successfully replicated the core components of the original Transformer[cite: 8, 9, 14, 17]:

  * [cite_start]**Multi-Head Attention**: 8 parallel heads with scaled dot-product attention[cite: 17].
  * [cite_start]**Encoder-Decoder Stack**: Full 6-layer stack with residual connections[cite: 17].
  * [cite_start]**Positional Encodings**: Standard sinusoidal geometric progression for handling sequence order[cite: 12].
  * **Embeddings**: Shared weights between input/output embeddings and the final linear projection.
  * [cite_start]**Regularization**: Residual Dropout ($P_{drop}=0.1$) and Label Smoothing ($\epsilon_{ls}=0.1$)[cite: 18].
  * [cite_start]**Optimizer**: Custom `NoamOpt` scheduler with linear warmup and inverse square root decay[cite: 31].

### 2\. Modern Improvements (Implemented)

  * [cite_start]**Pre-Layer Normalization (Pre-LN)**[cite: 32]:
      * [cite_start]**Problem:** The original Post-LN placement often leads to unstable gradients at initialization[cite: 30, 31].
      * [cite_start]**Solution:** We moved Layer Normalization to the *input* of the sublayers (inside the residual block)[cite: 32].
      * [cite_start]**Benefit:** Training is significantly more stable and robust to hyperparameter changes[cite: 36].
      * *Configuration:* Toggled via `layer_norm_mode="pre_ln"` (default) or `post_ln`.

## 📂 Project Structure

```text
ml-lab-improved/
├── models/
│   ├── model.py       # Main Transformer, Encoder, Decoder classes
│   ├── layers.py      # EncoderLayer and DecoderLayer
│   ├── blocks.py      # Attention, Pre-LN/Post-LN Sublayers, FeedForward
│   └── embeddings.py  # Standard Sinusoidal Positional Encoding
├── data/
│   ├── loader.py      # TextDataLoader for src-trg pairs
│   └── batches.py     # Masking and Batch object handling
├── utils/
│   ├── optimizer.py   # Noam Learning Rate Scheduler
│   ├── loss.py        # Label Smoothing Loss
│   └── tokenizer.py   # BPE Tokenizer wrapper
├── train.py           # Training loop (Synthetic Copy Task & Real World)
├── inference.py       # Beam Search & Greedy Decoding
└── verify_changes.py  # Unit tests for Architecture & Pre-LN logic
```

## 🛠️ Installation

1.  **Clone and Setup Environment:**
    You can use the provided PowerShell script to set up the virtual environment automatically:

    ```powershell
    ./setup.ps1
    ```

2.  **Manual Installation:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate
    pip install -r requirements.txt
    ```

## 🏃 Usage

### 1\. Run Verification Tests

Ensure all components (Weight Sharing, Pre-LN logic, Beam Search) are functioning correctly:

```bash
python verify_changes.py
```

### 2\. Train on Synthetic Data (Copy Task)

To demonstrate the model's ability to learn simple patterns:

```bash
python run_train.py
```

*This uses the default configuration in `config.json`.*

### 3\. Train on Real Data

To train on your own source/target text files:

```bash
python train.py path/to/source.txt path/to/target.txt
```

### 4\. Inference

Run inference using Beam Search on a trained model:

```bash
python inference.py
```

## 🗺️ Roadmap & Planned Improvements

Based on our [Implementation Progress Report](https://www.google.com/search?q=ml_progress_report.pdf), the following features are planned for the next release:

  * [cite_start]**[Planned] Rotary Positional Embeddings (RoPE)**: Replacing absolute sinusoidal encodings with relative positioning to improve generalization on variable sequence lengths[cite: 39, 42].
  * [cite_start]**[Planned] GeLU Activation**: Replacing `ReLU` with Gaussian Error Linear Units to avoid the "dying ReLU" problem[cite: 46, 48].
  * [cite_start]**[Planned] FlashAttention / Linear Attention**: Optimizing the attention mechanism from $O(n^2)$ to $O(n)$ complexity for longer sequence processing[cite: 24, 26].

## 📜 License

MIT License. See `LICENSE` for details.
