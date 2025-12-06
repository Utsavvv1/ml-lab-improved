import torch.nn as nn
import torch.nn.functional as F
import copy
from .blocks import MultiHeadedAttention, PositionwiseFeedForward, LayerNorm, clones
from .layers import EncoderLayer, DecoderLayer
from .embeddings import Embeddings, PositionalEncoding

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, layer_norm_mode="pre_ln"):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # position = PositionalEncoding(d_model, dropout) # Removed for RoPE
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, layer_norm_mode), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout, layer_norm_mode), N),
        nn.Sequential(Embeddings(d_model, src_vocab)), # Removed c(position)
        nn.Sequential(Embeddings(d_model, tgt_vocab)), # Removed c(position)
        Generator(d_model, tgt_vocab)
    )
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    # Weight Sharing
    # 1. Share src and tgt embeddings (assuming shared vocab or at least intended sharing)
    model.src_embed[0].lut.weight = model.tgt_embed[0].lut.weight
    # 2. Share tgt embeddings and generator weights
    model.generator.proj.weight = model.tgt_embed[0].lut.weight
    
    return model
