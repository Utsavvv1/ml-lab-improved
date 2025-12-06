from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class BPETokenizer:
    def __init__(self, vocab_size=5000):
        # Initialize BPE model
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # PAD=0, SOS=1, EOS=2, UNK=3 based on order
        self.special_tokens = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
        self.trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=self.special_tokens)
        
    def train(self, files):
        "Train the tokenizer on a list of files."
        print(f"Training tokenizer on {files}...")
        self.tokenizer.train(files, self.trainer)
        
    def encode(self, text):
        "Encode text to IDs."
        return self.tokenizer.encode(text).ids
        
    def decode(self, ids):
        "Decode IDs to text."
        return self.tokenizer.decode(ids)
        
    def save(self, path):
        self.tokenizer.save(path)
        
    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)
        
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
        
    @property
    def pad_token_id(self):
        return self.tokenizer.token_to_id("[PAD]")
        
    @property
    def sos_token_id(self):
        return self.tokenizer.token_to_id("[SOS]")
        
    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id("[EOS]")
