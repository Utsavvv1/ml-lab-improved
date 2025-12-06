# Transformer Implementation (Attention Is All You Need)

This repository contains a PyTorch implementation of the Transformer model from the paper "Attention Is All You Need" by Ashish Vaswani et al.

## Structure

The project is organized into the following modules:

- `models/`: Contains the core model architecture.
  - `model.py`: The main `Transformer` class and `make_model` helper.
  - `layers.py`: `EncoderLayer` and `DecoderLayer` definitions.
  - `blocks.py`: `MultiHeadedAttention`, `PositionwiseFeedForward`, `SublayerConnection`.
  - `embeddings.py`: `Embeddings` and `PositionalEncoding`.
- `utils/`: Utility functions for training.
  - `loss.py`: Loss computation including `LabelSmoothing`.
  - `optimizer.py`: The `NoamOpt` rate scheduling optimizer described in the paper.
- `data/`: Data handling.
  - `batches.py`: `Batch` object and synthetic data generation.
- `train.py`: Main training script to demonstrate the model on a copy task.

## Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training Demo**:
   ```bash
   python train.py
   ```

## Implementation Details

This implementation adheres strictly to the paper's design:
- **Optimizer**: Uses the specific `Noam` learning rate schedule with warmup.
- **Regularization**: Implements Label Smoothing `KLDivLoss` instead of standard CrossEntropy.
- **Architecture**: Faithful reproduction of the Multi-Head Attention and Position-wise Feed-Forward networks.

## Testing

Test your trained model using the inference script:

```bash
python inference.py
```

This will load the trained `model_final.pt` and run a few test sequences to demonstrate the model's ability to copy the input sequence.
