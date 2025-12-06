import os
import time
import json
import torch
import torch.nn as nn
from models import make_model
from data import data_gen, Batch, subsequent_mask
from utils import SimpleLossCompute, LabelSmoothing, NoamOpt


def run_epoch(data_iter, model, loss_compute, epoch_num, log_interval=50):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % log_interval == 1:
            elapsed = time.time() - start
            print(f"Epoch Step: {i} Loss: {loss / batch.ntokens:.6f} Tokens/sec: {tokens / elapsed:.2f}")
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           ys,
                           subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def train_copy_task(config=None):
    """Train the copy task. Accepts an optional config dict for reproducible runs.

    Example config keys:
      V: vocab size
      N: number of layers
      epochs: number of epochs
      batch: batch size
      nbatches: number of batches per epoch
      save_path: path to save model
      smoothing: label smoothing value
      log_interval: steps between logs
    """
    if config is None:
        config = {}

    V = config.get("V", 11)
    N = config.get("N", 2)
    epochs = config.get("epochs", 10)
    batch = config.get("batch", 30)
    nbatches = config.get("nbatches", 20)
    save_path = config.get("save_path", "models/weights/model_final.pt")
    smoothing = config.get("smoothing", 0.0)
    log_interval = config.get("log_interval", 50)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=smoothing)

    model = make_model(V, V, N=N)

    # Noam optimizer
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    print("Starting Training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        run_epoch(data_gen(V, batch, nbatches), model,
                  SimpleLossCompute(model.generator, criterion, model_opt), epoch, log_interval=log_interval)
        model.eval()

    print("Training Complete.")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Small inference example
    print("Running Inference Example (Copy Task)...")
    model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    print("Source:", src)
    pred = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
    print("Predicted:", pred)


if __name__ == "__main__":
    # Allow running with a simple JSON config file path via ENV or default config.json
    config_path = os.environ.get("ML_LAB_CONFIG", "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except Exception:
            cfg = None
    else:
        cfg = None
    train_copy_task(cfg)
