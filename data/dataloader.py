import os
import sentencepiece as spm
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset

# preprocessing 
class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_sp, tgt_sp):
        self.src_data = [line.strip() for line in open(src_file, 'r', encoding='utf-8')]
        self.tgt_data = [line.strip() for line in open(tgt_file, 'r', encoding='utf-8')]
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_line = self.src_data[idx]
        tgt_line = self.tgt_data[idx]
        src_tokens = self.src_sp.encode(src_line, out_type=int)
        tgt_tokens = self.tgt_sp.encode(tgt_line, out_type=int)
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)