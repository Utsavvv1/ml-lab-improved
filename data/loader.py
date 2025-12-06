import torch
from .batches import Batch

class TextDataLoader:
    def __init__(self, src_file, trg_file, tokenizer, batch_size=32, device='cpu'):
        self.src_file = src_file
        self.trg_file = trg_file
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        
    def __iter__(self):
        # Read files
        try:
            with open(self.src_file, 'r', encoding='utf-8') as f:
                src_lines = f.readlines()
            with open(self.trg_file, 'r', encoding='utf-8') as f:
                trg_lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: Data files not found: {self.src_file} or {self.trg_file}")
            return

        if len(src_lines) != len(trg_lines):
            print("Warning: Source and Target files have different lengths. Truncating to shorter.")
            min_len = min(len(src_lines), len(trg_lines))
            src_lines = src_lines[:min_len]
            trg_lines = trg_lines[:min_len]
            
        data = []
        sos_id = self.tokenizer.sos_token_id
        eos_id = self.tokenizer.eos_token_id
        
        for src, trg in zip(src_lines, trg_lines):
            s_ids = self.tokenizer.encode(src.strip())
            t_ids = self.tokenizer.encode(trg.strip())
            
            # Add SOS and EOS to target for Teacher Forcing / Prediction
            # Source usually doesn't need SOS/EOS strictly for Encoder, but EOS is good practice.
            # Transformer paper says: "We also use the usual learned positional embeddings..."
            # Usually strict: [SOS, ...seq..., EOS]
            
            # src: [s1, s2, ..., sn] -> Let's just padding processing handle it.
            # But we'll add EOS to src.
            # tgt: [SOS, t1, t2, ..., EOS]
            # Batch class splits tgt into inputs (:-1) and targets (1:)
            # So tgt MUST have SOS at start and EOS at end.
            
            # Check if tokenizer adds them? My wrapper doesn't yet. Manual add.
            
            # s_ids = [sos_id] + s_ids + [eos_id] # Optional for src
            # t_ids = [sos_id] + t_ids + [eos_id]
            
            # Let's add EOS to src, and SOS+EOS to tgt
            # src: [w1...wn, EOS]
            # tgt: [SOS, w1...wn, EOS]
            
            # If w1...wn is empty? Handle
            if not s_ids: s_ids = [self.tokenizer.tokenizer.token_to_id("[UNK]")]
            if not t_ids: t_ids = [self.tokenizer.tokenizer.token_to_id("[UNK]")]

            s_ids = s_ids # + [eos_id] # Simple src
            t_ids = [sos_id] + t_ids + [eos_id]
            
            data.append((s_ids, t_ids))
            
        # Basic bucket/sort to minimize padding
        data.sort(key=lambda x: len(x[0]))
        
        pad_id = self.tokenizer.pad_token_id
        
        for i in range(0, len(data), self.batch_size):
            chunk = data[i:i+self.batch_size]
            
            max_src = max(len(x[0]) for x in chunk)
            max_trg = max(len(x[1]) for x in chunk)
            
            src_tensor = torch.full((len(chunk), max_src), pad_id).long()
            trg_tensor = torch.full((len(chunk), max_trg), pad_id).long()
            
            for j, (s, t) in enumerate(chunk):
                src_tensor[j, :len(s)] = torch.tensor(s)
                trg_tensor[j, :len(t)] = torch.tensor(t)
                
            yield Batch(src_tensor.to(self.device), trg_tensor.to(self.device), pad=pad_id)
