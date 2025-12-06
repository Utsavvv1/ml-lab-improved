import torch
import sys
import os
import sacrebleu
from tqdm import tqdm
from models import make_model
from utils.tokenizer import BPETokenizer
from inference import beam_search
from data import subsequent_mask

def load_model(path, V):
    print(f"Loading model from {path}...")
    model = make_model(V, V, N=6) # Ensure N matches training (default 6 here, but training typically uses 6 or 2 based on copy task vs real)
    # The user might have trained with N=6 in train_real_world, so we stick to 6.
    # ideally we should load config. For now assuming N=6.
    
    try:
        model.load_state_dict(torch.load(path))
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying N=2 (Copy Task default) just in case...")
        model = make_model(V, V, N=2)
        model.load_state_dict(torch.load(path))
        
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model

def evaluate(model_path, src_file, ref_file, tokenizer_path="tokenizer.json", beam_size=5, max_len=100):
    # 1. Load Tokenizer
    tokenizer = BPETokenizer()
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        return
    tokenizer.load(tokenizer_path)
    V = tokenizer.get_vocab_size()
    
    # 2. Load Model
    model = load_model(model_path, V)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. Read Data
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f]
    with open(ref_file, 'r', encoding='utf-8') as f:
        ref_lines = [line.strip() for line in f]
        
    if len(src_lines) != len(ref_lines):
        print("Warning: Source and Reference files have different lengths.")
        min_len = min(len(src_lines), len(ref_lines))
        src_lines = src_lines[:min_len]
        ref_lines = ref_lines[:min_len]
        
    # 4. Generate Hypotheses
    hypotheses = []
    print(f"Evaluating on {len(src_lines)} sentences...")
    
    for src_line in tqdm(src_lines):
        # Prepare input
        ids = tokenizer.encode(src_line)
        src_tensor = torch.LongTensor([ids]).to(device)
        src_mask = torch.ones(1, 1, len(ids)).to(device)
        
        # Beam Search
        with torch.no_grad():
            out_seq = beam_search(model, src_tensor, src_mask, max_len=max_len, start_symbol=tokenizer.sos_token_id, beam_size=beam_size)
            
        # Decode
        # output is tensor [1, seq_len] or [seq_len] depending on beam search impl return
        # beam_search returns beam[0][0] which is 1xSeqLen
        out_ids = out_seq.flatten().tolist()
        
        # Remove SOS/EOS if present?
        # Typically tokenizer.decode handles it or we strip.
        # Let's simple decode.
        decoded = tokenizer.decode(out_ids)
        hypotheses.append(decoded)
        
    # 5. Calculate BLEU
    # sacrebleu expects list of hypotheses and list of references (refs must be list of lists if multiple refs per sentence)
    bleu = sacrebleu.corpus_bleu(hypotheses, [ref_lines])
    
    print("\n" + "="*30)
    print(f"BLEU Score: {bleu.score}")
    print("="*30)
    print("Example translations:")
    for i in range(min(5, len(hypotheses))):
        print(f"Src: {src_lines[i]}")
        print(f"Ref: {ref_lines[i]}")
        print(f"Hyp: {hypotheses[i]}")
        print("-" * 20)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python evaluate.py <model_path> <src_file> <ref_file> [tokenizer_path]")
    else:
        m_path = sys.argv[1]
        s_file = sys.argv[2]
        r_file = sys.argv[3]
        t_path = sys.argv[4] if len(sys.argv) > 4 else "tokenizer.json"
        evaluate(m_path, s_file, r_file, t_path)
