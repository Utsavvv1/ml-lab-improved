import torch
import os
from models import make_model, SublayerConnection
from inference import beam_search
from utils.tokenizer import BPETokenizer
from data.loader import TextDataLoader

def test_weight_sharing():
    print("Testing Weight Sharing...")
    model = make_model(100, 100, N=2)
    
    # Check 1: Src embedding and Tgt embedding weights
    assert id(model.src_embed[0].lut.weight) == id(model.tgt_embed[0].lut.weight)
    
    # Check 2: Tgt embedding and Generator weights
    assert id(model.generator.proj.weight) == id(model.tgt_embed[0].lut.weight)
    print("PASS: Weight Sharing Linked Correctly.")

def test_post_ln():
    print("Testing Post-LN...")
    # Pre-LN (default)
    sub = SublayerConnection(512, 0.1, mode="pre_ln")
    assert sub.mode == "pre_ln"
    
    # Post-LN
    sub_post = SublayerConnection(512, 0.1, mode="post_ln")
    assert sub_post.mode == "post_ln"
    
    # Pass data
    x = torch.randn(1, 512)
    sublayer = lambda x: x # Identity
    out = sub_post(x, sublayer)
    assert out.shape == x.shape
    print("PASS: Post-LN initialized and ran.")

def test_beam_search():
    print("Testing Beam Search...")
    model = make_model(11, 11, N=2)
    model.eval()
    
    src = torch.LongTensor([[1, 2, 3]])
    src_mask = torch.ones(1, 1, 3)
    
    # Just check it runs without error and returns scalar tensor (1, len)
    # or (1, len) ? beam_search returns (1, len) tensor from my implementation?
    # actually it returns the SEQUENCE tensor which is (1, len).
    try:
        out = beam_search(model, src, src_mask, max_len=5, start_symbol=1, beam_size=2)
        print("Beam Search Output shape:", out.shape)
        assert out.dim() == 2
        print("PASS: Beam Search ran.")
    except Exception as e:
        print(f"FAIL: Beam Search error: {e}")
        raise e

def test_flash_attention():
    print("Testing Flash Attention...")
    import torch.nn.functional as F
    if hasattr(F, 'scaled_dot_product_attention'):
        print("PASS: PyTorch version supports SDPA (FlashAttention).")
        # Could verify it runs
        q = torch.randn(1, 8, 32, 64)
        k = torch.randn(1, 8, 32, 64)
        v = torch.randn(1, 8, 32, 64)
        from models.blocks import attention
        # Mock dropout
        drop = torch.nn.Dropout(0.1)
        out, _ = attention(q, k, v, mask=None, dropout=drop)
        assert out.shape == q.shape
        print("PASS: Optimized attention ran successfully.")
    else:
        print("WARNING: PyTorch too old for SDPA.")

def test_rope():
    print("Testing RoPE...")
    from models.blocks import RotaryEmbedding, apply_rotary_pos_emb
    
    dim = 64
    rope = RotaryEmbedding(dim)
    
    # Create query [Batch, Head, Seq, Dim]
    q = torch.randn(1, 1, 10, dim)
    k = torch.randn(1, 1, 10, dim)
    
    cos, sin = rope(q, seq_len=10)
    assert cos.shape == (1, 1, 10, dim)
    
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    
    assert q_rot.shape == q.shape
    # Check that it actually changed
    assert not torch.allclose(q_rot, q)
    print("PASS: RoPE applied rotation.")

def test_gelu():
    print("Testing GeLU...")
    from models.blocks import PositionwiseFeedForward
    import torch.nn.functional as F
    
    d_model = 64
    d_ff = 128
    ff = PositionwiseFeedForward(d_model, d_ff)
    
    x = torch.randn(1, 10, d_model)
    out = ff(x)
    assert out.shape == x.shape
    # Check if F.gelu is actually being used (simple static analysis check or just runtime)
    # Runtime is fine.
    print("PASS: GeLU FeedForward ran successfully.")

def test_data_pipeline():
    print("Testing Data Pipeline...")
    # Create dummy files
    stop_train = False
    if not os.path.exists("test.src"):
        with open("test.src", "w") as f: f.write("hello world\nthis is a test\n")
        with open("test.trg", "w") as f: f.write("hallo welt\ndas ist ein test\n")
        stop_train = True
        
    tok = BPETokenizer(vocab_size=100)
    tok.train(["test.src", "test.trg"])
    assert tok.get_vocab_size() > 0
    
    loader = TextDataLoader("test.src", "test.trg", tok, batch_size=2)
    for batch in loader:
        print("Batch Src Shape:", batch.src.shape)
        print("Batch Trg Shape:", batch.trg.shape)
        assert batch.src.shape[0] <= 2
        break
        
    print("PASS: Data Pipeline loaded.")
    
    # Cleanup
    if stop_train:
        os.remove("test.src")
        os.remove("test.trg")
        # os.remove("tokenizer.json") # keep it or remove

if __name__ == "__main__":
    test_weight_sharing()
    test_post_ln()
    test_beam_search()
    test_beam_search()
    test_flash_attention()
    test_rope()
    test_gelu()
    test_data_pipeline()
    print("ALL TESTS PASSED.")
