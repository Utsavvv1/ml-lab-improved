import torch
import sys
import os
from models import make_model
from utils.tokenizer import BPETokenizer
from inference import beam_search

def load_translator(model_path="model_real.pt", tokenizer_path="tokenizer.json"):
    """
    Loads the necessary components for translation.
    
    Returns:
        model: The trained PyTorch model (in eval mode).
        tokenizer: The BPE tokenizer.
    """
    # 1. Load Tokenizer
    tokenizer = BPETokenizer()
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        return None, None
    tokenizer.load(tokenizer_path)
    V = tokenizer.get_vocab_size()
    
    # 2. Load Model
    print(f"Loading model from {model_path}...")
    try:
        # Try Loading Real Model (N=6)
        model = make_model(V, V, N=6)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except:
        try:
             # Try Dummy Model (N=2) - Fallback for small tests
             print("N=6 failed, trying N=2...")
             model = make_model(V, V, N=2, d_model=128, d_ff=512, h=4)
             model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        except Exception as e:
             print(f"Failed to load model: {e}")
             return None, None

    model.eval()
    return model, tokenizer

def translate(sentence, model, tokenizer):
    """
    Translates a single sentence string using the model.
    """
    device = "cpu" # Inference on CPU is fine for single sentences
    
    # Encode user input to token IDs
    ids = tokenizer.encode(sentence)
    src = torch.LongTensor([ids]).to(device)
    src_mask = torch.ones(1, 1, len(ids)).to(device)
    
    # Decode using Beam Search
    # Note: If the model is not well-trained (e.g. 1 epoch dummy), it repeats tokens.
    # Increasing max_len just makes it repeat longer.
    # We will try a smaller max_len for testing.
    out_seq = beam_search(model, src, src_mask, max_len=30, start_symbol=tokenizer.sos_token_id, beam_size=3)
    
    # Convert token IDs back to text
    out_ids = out_seq.flatten().tolist()
    return tokenizer.decode(out_ids)

def main():
    str_sentence = "A dog runs in the snow."
    
    if len(sys.argv) > 1:
        str_sentence = sys.argv[1]
        
    model, tokenizer = load_translator()
    if model is None: return
    
    # Interactive mode if no arg provided
    if len(sys.argv) <= 1:
        print("Enter a sentence to translate (type 'q' to quit):")
        while True:
            text = input("> ")
            if text.lower() == 'q': break
            result = translate(text, model, tokenizer)
            print(f"Translation: {result}")
    else:
        # One-shot
        result = translate(str_sentence, model, tokenizer)
        print(f"Input: {str_sentence}")
        print(f"Translation: {result}")

if __name__ == "__main__":
    main()
