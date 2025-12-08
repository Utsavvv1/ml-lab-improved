import torch
import sys
import os
from models.model import make_model
from utils.tokenizer import BPETokenizer
from inference import beam_search

def load_real_model(model_path="model_final.pt", tokenizer_path="tokenizer.json"):
    # 1. Load Tokenizer
    tokenizer = BPETokenizer()
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        return None, None
    tokenizer.load(tokenizer_path)
    V = tokenizer.get_vocab_size()
    
    # 2. Load Model
    print(f"Loading real weights from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} does not exist.")
        return None, None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard architecture params matching the report/training
    model = make_model(src_vocab=V, tgt_vocab=V, N=6, d_model=512, d_ff=2048, h=8)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load state dict: {e}")
        return None, None

    model.to(device)
    model.eval()
    return model, tokenizer, device

def translate_sentence(sentence, model, tokenizer, device):
    model.eval()
    ids = tokenizer.encode(sentence)
    src = torch.LongTensor([ids]).to(device)
    src_mask = torch.ones(1, 1, len(ids)).to(device)
    
    # Beam search equivalent to what is used in inference.py
    # We use a small beam size for speed in this demo
    out_seq = beam_search(model, src, src_mask, max_len=60, start_symbol=tokenizer.sos_token_id, beam_size=3)
    
    out_ids = out_seq.flatten().tolist()
    return tokenizer.decode(out_ids)

def main():
    str_sentence = "A dog runs in the snow."
    if len(sys.argv) > 1:
        str_sentence = " ".join(sys.argv[1:])
        
    model, tokenizer, device = load_real_model()
    if model is None: return
    
    if len(sys.argv) <= 1:
        print("Enter a sentence to translate (type 'q' to quit):")
        while True:
            try:
                text = input("> ")
                if text.lower() == 'q': break
                
                result = translate_sentence(text, model, tokenizer, device)
                print(f"Real Model Translation: {result}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error during translation: {e}")
    else:
        result = translate_sentence(str_sentence, model, tokenizer, device)
        print(f"Input: {str_sentence}")
        print(f"Real Model Translation: {result}")

if __name__ == "__main__":
    main()
