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

def train_real_world(src_file, trg_file, tokenizer_path="tokenizer.json", vocab_size=5000, epoch_count=10):
    print(f"Starting Real World Training with {src_file} -> {trg_file}")
    
    # 1. Train or Load Tokenizer
    from utils.tokenizer import BPETokenizer
    tokenizer = BPETokenizer(vocab_size)
    import os
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer.load(tokenizer_path)
    else:
        print("Training Tokenizer...")
        tokenizer.train([src_file, trg_file])
        tokenizer.save(tokenizer_path)
        
    V = tokenizer.get_vocab_size()
    print(f"Vocabulary Size: {V}")
    
    # 2. Model
    # Note: You might want to tune d_model etc for real tasks
    model = make_model(V, V, N=6) 
    
    # 3. Optimizer
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
            
    # 4. Criterion
    pad_idx = tokenizer.pad_token_id
    criterion = LabelSmoothing(size=V, padding_idx=pad_idx, smoothing=0.1)
    
    if torch.cuda.is_available():
        model.cuda()
        device = "cuda"
    else:
        device = "cpu"
    
    # 5. Loop
    from data.loader import TextDataLoader
    for epoch in range(epoch_count):
        print(f"Epoch {epoch}")
        model.train()
        loader = TextDataLoader(src_file, trg_file, tokenizer, batch_size=32, device=device)
        run_epoch(loader, model, SimpleLossCompute(model.generator, criterion, model_opt), epoch)
        model.eval()
        # Validation could go here
        
    torch.save(model.state_dict(), "model_real.pt")
    print("Model saved to model_real.pt")

if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) > 2 and os.path.exists(sys.argv[1]):
        # Usage: python train.py data/train.de data/train.en
        train_real_world(sys.argv[1], sys.argv[2])
    else:
        print("No real data files provided or found. Running Synthetic Copy Task...")
        train_copy_task()
