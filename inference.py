import torch
from models import make_model
from data import subsequent_mask

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

def beam_search(model, src, src_mask, max_len, start_symbol, beam_size=5):
    memory = model.encode(src, src_mask)
    start_seq = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # beam = [(sequence, log_prob_sum)]
    beam = [(start_seq, 0.0)]
    
    # Paper uses alpha = 0.6 for length penalty
    alpha = 0.6

    for i in range(max_len-1):
        candidates = []
        for seq, log_prob_sum in beam:
            if seq[0, -1] == 2: # Stop if EOS (assuming 2 is EOS, adjust if different)
                 candidates.append((seq, log_prob_sum))
                 continue
                 
            # Expand
            out = model.decode(memory, src_mask, 
                               seq, 
                               subsequent_mask(seq.size(1)).type_as(src.data))
            prob = model.generator(out[:, -1])
            
            # Get top k for this beam
            topk_probs, topk_indices = torch.topk(prob, beam_size, dim=1)
            
            for k in range(beam_size):
                next_word = topk_indices[0, k].item()
                prob_score = topk_probs[0, k].item()
                
                new_seq = torch.cat([seq, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
                new_log_prob_sum = log_prob_sum + prob_score
                candidates.append((new_seq, new_log_prob_sum))
        
        # Sort by LENGTH PENALIZED score: score / ((5 + len) / 6) ^ alpha
        def get_score(candidate):
            seq, lp_sum = candidate
            length = seq.size(1)
            penalty = ((5 + length) / 6) ** alpha
            return lp_sum / penalty

        candidates.sort(key=get_score, reverse=True)
        beam = candidates[:beam_size]
        
        # Break if all beams ended (optional optimization)
        if all(c[0][0, -1] == 2 for c in beam):
            break
            
    return beam[0][0]

def load_model(path="model_final.pt", V=11):
    print(f"Loading model from {path}...")
    # Re-initialize the model structure with the same parameters as training
    # Important: Must match the weight sharing and structure of saved model if loaded
    # For now assuming compatible loading or we'll retrain
    model = make_model(V, V, N=2)
    # Allow partial load if strict=False or handle mismatches if weights were shared differently in saved model
    # But user wants us to implement these things, likely for a NEW run.
    try:
        model.load_state_dict(torch.load(path))
    except Exception as e:
        print(f"Warning: Could not load state dict cleanly: {e}")
        print("Initializing random weights for demonstration.")
        
    model.eval()
    return model

def inference(src_seq):
    # Prepare model
    V = 11
    model = load_model()
    
    # Prepare input
    src = torch.LongTensor([src_seq])
    src_mask = torch.ones(1, 1, src.size(1))
    
    print(f"\nSource Sequence: {src_seq}")
    print("Generating with Beam Search...")
    
    pred = beam_search(model, src, src_mask, max_len=len(src_seq), start_symbol=1, beam_size=3)
    
    print(f"Predicted Output: {pred.flatten().tolist()}")

if __name__ == "__main__":
    # Example test cases
    test_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    inference(test_seq)
    
    test_seq_2 = [1, 5, 4, 3, 2, 1, 9, 8, 7, 6]
    inference(test_seq_2)
