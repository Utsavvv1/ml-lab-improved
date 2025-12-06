import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout, mode="pre_ln"):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.mode = mode

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if self.mode == "pre_ln":
            # Pre-LN: Apply normalization BEFORE the sublayer.
            # This stabilizes training by keeping the residual path clean (x + f(norm(x))).
            # Used in GPT-2/3 and most modern Transformers.
            return x + self.dropout(sublayer(self.norm(x)))
        elif self.mode == "post_ln":
            # Post-LN: Apply normalization AFTER the residual connection.
            # Matches the original "Attention Is All You Need" paper.
            # x = norm(x + f(x))
            return self.norm(x + self.dropout(sublayer(x)))

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies Rotary Position Embedding to the query and key tensors.
       q, k: [Batch, Head, SeqLen, HeadDim]
       cos, sin: [1, 1, SeqLen, HeadDim] OR [Batch, 1, SeqLen, HeadDim] etc.
    """
    # Just in case cos/sin need reshaping for broadcasting
    # usually cos/sin are (1, 1, Seq, Dim)
    # q is (Batch, Head, Seq, Dim)
    
    # Check if we need to slice cos/sin to sequence length
    seq_len = q.shape[2]
    if cos.shape[2] < seq_len:
         # Dynamic resizing is needed or error.
         # For simplicity, we assume pre-computed cache is large enough or we recompute.
         # Recomputing on the fly is safer for variable length
         pass # Handled in MHA
         
    # Slice to current seq len
    cos = cos[:, :, :seq_len, :]
    sin = sin[:, :, :seq_len, :]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but common implementation: cat freq with itself
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [Batch, Head, Seq, Dim]
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :].to(x.device)
            self.sin_cached = emb.sin()[None, None, :, :].to(x.device)
            
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # Optimized implementation using PyTorch SDPA (FlashAttention compatible)
    if hasattr(F, 'scaled_dot_product_attention'):
        # SDPA expects dropout probability (p) not the module
        p = 0.0
        if dropout is not None:
             p = dropout.p
             
        # Handling Mask
        # SDPA expects:
        #   - None (no mask)
        #   - Boolean mask (True = attend, False = ignore) OR (True = ignore, False = attend) depending on version, 
        #     but usually for causal it supports is_causal=True.
        #   - Additive mask (float) where -inf is ignore.
        
        # Our mask is currently 1 for valid, 0 for invalid (padding).
        # We also have subsequent_mask which is tril.
        
        # Simple fix for now: Convert 1/0 mask to boolean mask for SDPA? 
        # Actually SDPA supports attn_mask argument. 
        # It must be broadcastable.
        
        # Warning: For "Future Mask" (subsequent), SDPA has is_causal=True.
        # But our mask argument combines padding mask AND future mask.
        
        # Let's try passing the mask directly?
        # If mask is 0 (ignore) and 1 (keep), we need to ensure SDPA understands it.
        # PyTorch SDPA docs: "Binary mask ... True indicates that the corresponding position is not allowed to attend." (Wait, check docs).
        # Actually docs say: "attn_mask: ... shape (L, S) or (N, num_heads, L, S). 
        # If boolean, True = ignore? No, typically boolean mask where True=Keep is common in some libs but Torch might be opposite.
        # Let's stick to the additive mask approach which is safest across versions if we are unsure.
        # Our existing code did: scores.masked_fill(mask == 0, -1e9)
        # So mask==1 is Keep.
        
        # Let's just use the manual fallback if mask is complex, OR try to convert.
        # Actually, let's just do the manual scaled dot product if we want to be 100% safe with existing masks,
        # BUT the user specifically asked for FlashAttention.
        
        # Let's rely on F.scaled_dot_product_attention to handle the mask if possible.
        # If we pass a boolean mask, we need to be careful.
        # If we pass float mask, it works.
        
        # If mask is provided:
        if mask is not None:
             # Our mask is 1 (keep), 0 (ignore).
             # SDPA with float mask adds it to attention scores. 
             # So we need 0 for keep, -inf for ignore.
             # Convert:
             if mask.dtype != torch.float:
                 # Create additive mask
                 # (1 -> 0, 0 -> -inf)
                 # We can just use the manual logic which is reliable, 
                 # OR convert mask. 
                 pass
             
    # Fallback / Original Implementation (Reference)
    # We will try to use SDPA if dimensions allow and mask is handled.
    
    # Actually, to properly support FlashAttention, we should convert the mask.
    if hasattr(F, 'scaled_dot_product_attention'):
         dropout_p = dropout.p if dropout is not None else 0.0
         is_causal = False # We handle causality via mask
         
         # Convert mask to additive if present
         attn_mask = None
         if mask is not None:
             # mask: [Batch, 1, Seq, Seq]
             # We need to construct a mask that SDPA likes.
             # For padding mask (1=valid, 0=invalid), we want 0 for valid, -inf for invalid.
             # Check if mask is boolean:
             if mask.dtype == torch.bool or mask.dtype == torch.uint8 or mask.dtype == torch.long:
                  # Create float mask
                  attn_mask = torch.zeros_like(mask, dtype=query.dtype)
                  attn_mask.masked_fill_(mask == 0, float("-inf"))
             else:
                  # Assume it is already suitable or we fallback
                  attn_mask = mask

         return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal), None
         
    # Legacy Fallback
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(self.d_k)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
             
        # 2) Apply RoPE
        # query, key: [Batch, Head, Seq, Dim]
        cos, sin = self.rotary_emb(value, seq_len=query.size(2)) 
        # Note: If key length diff from query length (cross attn), usage is nuanced.
        # But typically RoPE is critical for *Self Attention*.
        # For Cross Attention (Decoder-Encoder), usually:
        #   - Query (Decoder) has Decoder position
        #   - Key (Encoder) has Encoder position
        #   - They don't align in relative distance same way.
        # Common practice: Apply RoPE to Self-Attn, maybe NO RoPE to Cross-Attn or handle carefully.
        # Here we apply to both if q/k lengths match context, or we re-generate for k.
        
        # Wait, self.rotary_emb(value, seq_len=...) generates up to seq_len.
        # Query len and Key len might differ in Cross Attention!
        # Decoder: Query is Tgt (len M), Key is Src (len N).
        # We need cos/sin for Query and cos/sin for Key separately?
        
        # Correct handling:
        # Self Attn: Q, K have same positions.
        # Cross Attn: Q has pos 0..M, K has pos 0..N.
        # Standard RoPE typically applies to Self-Attention.
        
        # Let's check typical RoPE implementations (e.g. LLaMA). 
        # It is applied in self-attention.
        # Does Transformer generic MHA handle Cross? Yes.
        # If Cross-Attn, do we rotate?
        # Many papers (e.g. "RoPE for Cross Attention") suggest it's tricky.
        # Usually, Absolute Pos Emb is removed, so we NEED some pos info in Cross Attn.
        # But standard RoPE is relative.
        # For now, let's assume RoPE is primarily for Self-Attention (where Q=K=V inputs are same source)
        # OR we generate cos/sin for max(q_len, k_len) and slice?
        
        # Let's generate for Q and K independently based on their lengths:
        cos_q, sin_q = self.rotary_emb(query, seq_len=query.size(2))
        cos_k, sin_k = self.rotary_emb(key, seq_len=key.size(2))
        
        query, _ = apply_rotary_pos_emb(query, query, cos_q, sin_q) # Dummy 2nd arg
        key, _ = apply_rotary_pos_emb(key, key, cos_k, sin_k) 

        # 3) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 4) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))
