import torch

def create_mask(src, tgt, pad_token):
    """
    Input: 
        - src: (seq_len, bs)
        - tgt: (seq_len, bs)
        - pad_token: int 
    
    Output: 
        - src_mask: mask out certain position in the source sequence, usually zeros 
        - tgt_mask: prevent the decoder from seeing future tokens during training (1 masked, 0 visible)
        - src_padding_mask: mask out padding tokens in the source sequenes to ignore padding tokens
        - tgt_padding_mask: mask out padding tokens in the target sequences
        - memory_mask: mask out certain positions in the source sequence while decoding
    """
    
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

    src_padding_mask = (src == pad_token).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_token).transpose(0, 1)

    # memory_mask:  디코더가 인코더의 출력(메모리)을 참조할 때 특정 위치를 마스킹하는 데 사용됨
    # 대부분의 경우 memory_mask는 모든 위치를 참조할 수 있도록 설정되며, 이는 디폴트로 설정된 마스크입니다. 
    # Create memory mask to mask out padding tokens in the source sequences
    memory_mask = src_padding_mask.unsqueeze(1).expand(-1, tgt_seq_len, -1).transpose(0, 1)
    # print("src_mask:", src_mask.shape)
    # print("tgt_mask:", tgt_mask.shape)
    # print("src_padding_mask:", src_padding_mask.shape)
    # print("tgt_padding_mask:", tgt_padding_mask.shape)
    # print("memory_mask:", memory_mask.shape) #(tgt_seq_len, bs, src_seq_len)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_mask

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
