import os
import time
import json
import torch
import torch.nn as nn
from models import make_model
from data import data_gen, Batch, subsequent_mask
from utils import SimpleLossCompute, LabelSmoothing, NoamOpt


def run_epoch(data_iter, model, loss_compute, epoch_num, log_interval=50):
    """
    Standard Training and Logging Function.
    
    Args:
        data_iter: Iterator returning batches of data.
        model: The Transformer model to train or evaluate.
        loss_compute: Function to calculate loss and handle backpropagation.
        epoch_num: Current epoch number (for display).
        log_interval: How often to print training stats (batches).
        
    Returns:
        Average loss across the epoch.
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    
    # Iterate over all batches in the dataset
    for i, batch in enumerate(data_iter):
        # Forward pass: compute the model output given source and target
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
                            
        # Compute loss: compare output with actual target (trg_y)
        # This function also handles the backward pass and optimizer step if training
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        
        # Logging
        if i % log_interval == 1:
            elapsed = time.time() - start
            print(f"Epoch Step: {i} Loss: {loss / batch.ntokens:.6f} Tokens/sec: {tokens / elapsed:.2f}")
            start = time.time()
            tokens = 0
            
    return total_loss / total_tokens


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Greedy decoding for inference. 
    Selects the word with highest probability at each step.
    
    Args:
        model: Trained Transformer model.
        src: Source sequence tensor.
        src_mask: Mask for source sequence.
        max_len: Maximum length of generated sequence.
        start_symbol: ID of the [SOS] token to start generation.
    """
    # 1. Encode the source sequence once
    memory = model.encode(src, src_mask)
    
    # 2. Initialize target sequence with start symbol
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    # 3. Autoregressive decoding loop
    for i in range(max_len - 1):
        # Decode the current sequence using the encoded memory
        out = model.decode(memory, src_mask,
                           ys,
                           subsequent_mask(ys.size(1)).type_as(src.data))
                           
        # Get probabilities for the last token
        prob = model.generator(out[:, -1])
        
        # Pick the single most likely next token (Greedy)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        
        # Append next word to the sequence
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
    """
    Train on real-world text files.
    
    Steps:
    1. Train/Load a BPE tokenizer on the data.
    2. Initialize the Transformer model.
    3. Run the training loop for specified epochs.
    4. Save checkpoints and final model.
    """
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
    
    # 2. Model Initialization
    # N=6 is standard for "Attention is All You Need", but heavy for CPUs.
    # Note: You might want to tune d_model etc for real tasks
    model = make_model(V, V, N=6) 
    
    # 3. Optimizer
    # NoamOpt implements the warm-up and decay schedule from the paper.
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
            
    # 4. Criterion (Loss Function)
    # LabelSmoothing helps prevent the model from becoming too confident.
    pad_idx = tokenizer.pad_token_id
    criterion = LabelSmoothing(size=V, padding_idx=pad_idx, smoothing=0.1)
    
    if torch.cuda.is_available():
        model.cuda()
        device = "cuda"
    else:
        device = "cpu"
    
    # 5. Training Loop
    from data.loader import TextDataLoader
    for epoch in range(epoch_count):
        print(f"Epoch {epoch}")
        model.train() # Set model to training mode (enables dropout)
        
        # Loader handles reading files, tokenizing, and creating batches
        loader = TextDataLoader(src_file, trg_file, tokenizer, batch_size=32, device=device)
        
        run_epoch(loader, model, SimpleLossCompute(model.generator, criterion, model_opt), epoch)
        
        print(f"Saving checkpoint for epoch {epoch}...")    
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")
        # Validation could go here
        
    torch.save(model.state_dict(), "model_real.pt")
    print("Model saved to model_real.pt")

def train_dummy_small(src_file, trg_file, tokenizer_path="tokenizer.json"):
    print("Starting Dummy Small Training for BLEU check...")
    
    # 1. Train Tokenizer
    from utils.tokenizer import BPETokenizer
    tokenizer = BPETokenizer(vocab_size=2000) # Small vocab
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    else:
        tokenizer.train([src_file, trg_file])
        tokenizer.save(tokenizer_path)
    
    V = tokenizer.get_vocab_size()
    
    # 2. Small Model (For fast CPU training)
    # N=2 layers, d_model=128, h=4
    # The current make_model supports: src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, layer_norm_mode="pre_ln"
    
    model = make_model(V, V, N=2, d_model=128, d_ff=512, h=4)
    
    criterion = LabelSmoothing(size=V, padding_idx=tokenizer.pad_token_id, smoothing=0.1)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    
    # 3. Quick loop
    from data.loader import TextDataLoader
    # Train for just 1 epoch on small batch
    for epoch in range(1):
        print(f"Epoch {epoch}")
        model.train()
        loader = TextDataLoader(src_file, trg_file, tokenizer, batch_size=16, device="cpu")
        # Run only a few batches just to ensure it saves something non-random
        run_epoch(loader, model, SimpleLossCompute(model.generator, criterion, model_opt), epoch)
        
    torch.save(model.state_dict(), "model_dummy.pt")
    print("Dummy model saved to model_dummy.pt")

if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) > 1 and sys.argv[1] == "dummy":
        if len(sys.argv) > 3:
             train_dummy_small(sys.argv[2], sys.argv[3])
        else:
             print("Usage: python train.py dummy <src> <trg>")
    elif len(sys.argv) > 2 and os.path.exists(sys.argv[1]):
        # Usage: python train.py data/train.de data/train.en
        train_real_world(sys.argv[1], sys.argv[2])
    else:
        print("No real data files provided or found. Running Synthetic Copy Task...")
        train_copy_task()
