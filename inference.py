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

def load_model(path="model_final.pt", V=11):
    print(f"Loading model from {path}...")
    # Re-initialize the model structure with the same parameters as training
    model = make_model(V, V, N=2)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def inference(src_seq):
    # Prepare model
    V = 11
    model = load_model()
    
    # Prepare input
    # Ensure src_seq starts with 1 if it doesn't already, or handle as per logic (here we assume user gives raw numbers)
    # The training data always had 1 at index 0, but user might just want to copy [2,3,4]
    # For consistency with the copy task where index 0 is always 1 (start token/padding/placeholder in this context):
    
    # Let's just assume we pad with 1 at the start if not present, or just pass as is.
    # The training logic forced data[:, 0] = 1.
    
    src = torch.LongTensor([src_seq])
    src_mask = torch.ones(1, 1, src.size(1))
    
    print(f"\nSource Sequence: {src_seq}")
    print("Generating...")
    
    pred = greedy_decode(model, src, src_mask, max_len=len(src_seq), start_symbol=1)
    
    print(f"Predicted Output: {pred.flatten().tolist()}")

if __name__ == "__main__":
    # Example test cases
    test_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    inference(test_seq)
    
    test_seq_2 = [1, 5, 4, 3, 2, 1, 9, 8, 7, 6]
    inference(test_seq_2)
