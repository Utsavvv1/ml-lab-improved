import time
import torch
import torch.nn as nn
from models import make_model
from data import data_gen, Batch, subsequent_mask
from utils import SimpleLossCompute, LabelSmoothing, NoamOpt, get_std_opt

def run_epoch(data_iter, model, loss_compute, epoch_num):
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
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           ys, 
                           subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def train_copy_task():
    # Train the simple copy task.
    V = 11
    # Use LabelSmoothing as in the paper, though strictly speaking 
    # for this simple task, NLLLoss is fine. 
    # But for "correctness" regarding the User's strict request, we include it.
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    
    model = make_model(V, V, N=2)
    
    # Use the Noam Optimizer from the paper
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    
    print("Starting Training...")
    for epoch in range(10):
        print(f"Epoch {epoch}")
        model.train()
        run_epoch(data_gen(V, 30, 20), model, 
                  SimpleLossCompute(model.generator, criterion, model_opt), epoch)
        model.eval()
        # evaluation can be added here
    
    print("Training Complete.")
    torch.save(model.state_dict(), "model_final.pt")
    print("Model saved to model_final.pt")
    
    print("Running Inference Example (Copy Task)...")
    model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    print("Source:", src)
    pred = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
    print("Predicted:", pred)

if __name__ == "__main__":
    train_copy_task()
