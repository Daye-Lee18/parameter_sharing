# # # https://github.com/facebookresearch/fairseq/blob/main/fairseq/tasks/translation.py
# # # line 126
# # # https://github.com/facebookresearch/fairseq/blob/main/fairseq/tasks/translation.py#L291
# # # line 309 

# import os
# import sentencepiece as spm
# import torch
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
# from torch.utils.data import DataLoader, Dataset

# # preprocessing 
# class TranslationDataset(Dataset):
#     def __init__(self, src_file, tgt_file, src_sp, tgt_sp):
#         self.src_data = [line.strip() for line in open(src_file, 'r', encoding='utf-8')]
#         self.tgt_data = [line.strip() for line in open(tgt_file, 'r', encoding='utf-8')]
#         self.src_sp = src_sp
#         self.tgt_sp = tgt_sp

#     def __len__(self):
#         return len(self.src_data)

#     def __getitem__(self, idx):
#         src_line = self.src_data[idx]
#         tgt_line = self.tgt_data[idx]
#         src_tokens = self.src_sp.encode(src_line, out_type=int)
#         tgt_tokens = self.tgt_sp.encode(tgt_line, out_type=int)
#         return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

import torch
from torch.utils.data import DataLoader, Dataset
import os
import linecache

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.src_length = sum(1 for _ in open(src_file, 'r', encoding='utf-8'))
        self.tgt_length = sum(1 for _ in open(tgt_file, 'r', encoding='utf-8'))
        assert self.src_length == self.tgt_length, "Source and target files must have the same number of lines"

    def __len__(self):
        return self.src_length

    def __getitem__(self, idx):
        src_line = linecache.getline(self.src_file, idx + 1).strip().split()
        tgt_line = linecache.getline(self.tgt_file, idx + 1).strip().split()
        src_tokens = list(map(int, src_line))
        tgt_tokens = list(map(int, tgt_line))
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=3)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=3)
    return src_batch, tgt_batch

def get_dataloader(data_path, src_lang, tgt_lang, split, batch_size=32, num_workers=0):
    src_file = os.path.join(data_path, f'{split}.{src_lang}.bpe')
    tgt_file = os.path.join(data_path, f'{split}.{tgt_lang}.bpe')
    dataset = TranslationDataset(src_file, tgt_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    return dataloader
