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
    # beam = [(sequence_tensor, score)]
    start_seq = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    beam = [(start_seq, 0.0)]
    
    for i in range(max_len-1):
        candidates = []
        for seq, score in beam:
            # Expand
            out = model.decode(memory, src_mask, 
                               seq, 
                               subsequent_mask(seq.size(1)).type_as(src.data))
            prob = model.generator(out[:, -1])
            # prob is log_softmax
            
            # Get top k for this beam
            topk_probs, topk_indices = torch.topk(prob, beam_size, dim=1)
            
            for k in range(beam_size):
                next_word = topk_indices[0, k].item()
                prob_score = topk_probs[0, k].item()
                
                new_seq = torch.cat([seq, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
                new_score = score + prob_score
                candidates.append((new_seq, new_score))
        
        # Sort all candidates and prune to beam_size
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_size]
        
    # Return the best sequence
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
