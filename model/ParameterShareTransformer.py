import torch
import torch.nn as nn

class Encoder(nn.TransformerEncoder):
    def __init__(
        self,
        d_model=512,
        nhead=16,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        num_unique_layers=3,
        num_total_layers=6,
        mode="cycle_rev",
        norm=False,
    ):  
        # https://github.com/takase/share_layer_params/blob/main/fairseq/fairseq/models/transformer.py#L275
        # TransformerEncoder(), ,,, code 참고 
        assert mode in {"sequence", "cycle", "cycle_rev"}
        quotient, remainder = divmod(num_total_layers, num_unique_layers)
        assert remainder == 0 

        if mode == "cycle_rev":
            assert quotient == 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )

        super().__init__(encoder_layer, num_layers=num_unique_layers, norm=norm)
        self.N = num_total_layers
        self.M = num_unique_layers
        self.mode = mode
        self.norm = nn.LayerNorm(d_model) if norm else None
    
    def forward(self, x, mask=None, src_key_padding_mask=None, verbose=False):
        """
        Input: 
            - x (Tensor): input sequence embedding (bs, seq_len, d_model)
            - mask (Optional[Tensor]): future token masking (seq_len, seq_len)
            - src_key_padding_mask (Optional[Tensor]): indicating the padding token's positions with True value (bs, seq_len)
            - verbose (bool): for debugging
        
        Output:
            x (Tensor): encoded output (batch_size, seq_len, d_model)
        """
        for enc_i in range(self.N):
            if self.mode == "sequence":
                enc_i = enc_i // (self.N // self.M)
            elif self.mode == "cycle":
                enc_i = enc_i % self.M
            elif enc_i > (self.N - 1) / 2:
                enc_i = self.N - i - 1
            if verbose:
                print(f"layer {enc_i}")
            x = self.layers[enc_i](x, mask, src_key_padding_mask)
        if self.norm is not None:
            x = self.norm(x)
        
        x = x.transpose(0, 1)  # Change back to (batch_size, seq_len, d_model)
        return x               

class Decoder(nn.TransformerDecoder):
    def __init__(
        self,
        d_model=512,
        nhead=16,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        num_unique_layers=3,
        num_total_layers=6,
        mode="cycle_rev",
        norm=False,
    ):
        assert mode in {"sequence", "cycle", "cycle_rev"}
        quotient, remainder = divmod(num_total_layers, num_unique_layers)
        assert remainder == 0

        if mode == "cycle_rev":
            assert quotient == 2

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )

        super().__init__(decoder_layer, num_layers=num_unique_layers, norm=norm)
        self.N = num_total_layers
        self.M = num_unique_layers
        self.mode = mode
        self.norm = nn.LayerNorm(d_model) if norm else None

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, verbose=False):

        """
        Input: 
            - tgt (Tensor): target sequence embedding (batch_size, tgt_len, d_model)
            - memory (Tensor): encoder output ( batch_size, seq_len, d_model)
            - tgt_mask (Optional[Tensor]): target sequence mask preventing the model from seeing the future (tgt_len, tgt_len)
            - memory_mask (Optional[Tensor]): memory mask (tgt_len, seq_len)
            - tgt_key_padding_mask (Optional[Tensor]): target padding mask (batch_size, tgt_len)
            - memory_key_padding_mask (Optional[Tensor]): memory padding mask (batch_size, seq_len)
            - verbose (bool): for debugging
        
        Output: 
            - tgt (Tensor): decoded output for a target language (batch_size, tgt_len, d_model)
        """
        
        for dec_i in range(self.N):
            if self.mode == "sequence":
                dec_i = dec_i // (self.N // self.M)
            elif self.mode == "cycle":
                dec_i = dec_i % self.M
            elif dec_i > (self.N - 1) / 2:
                dec_i = self.N - dec_i - 1
            if verbose:
                print(f"layer {dec_i}")
            tgt = self.layers[dec_i](tgt, memory, tgt_mask, memory_mask,
                                     tgt_key_padding_mask, memory_key_padding_mask)
        if self.norm is not None:
            tgt = self.norm(tgt) 

        tgt = tgt.transpose(0, 1)  # Change back to (batch_size, tgt_len, d_model)
        return tgt

class ParameterShareTransformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=16,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        num_unique_encoder_layers=3,
        num_unique_decoder_layers=3,
        mode="cycle_rev",
    ):
        super().__init__()
        self.encoder = Encoder(
            d_model, nhead, dim_feedforward, dropout, activation,
            num_unique_encoder_layers, num_encoder_layers, mode, norm=True
        )
        self.decoder = Decoder(
            d_model, nhead, dim_feedforward, dropout, activation,
            num_unique_decoder_layers, num_decoder_layers, mode, norm=True
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        """
        Input:
            src (Tensor): source sequence (batch_size, src_len, d_model)
            tgt (Tensor): target sequence (batch_size, tgt_len, d_model)
            src_mask (Optional[Tensor]): source mask (src_len, src_len)
            tgt_mask (Optional[Tensor]): target mask (tgt_len, tgt_len)
            memory_mask (Optional[Tensor]): memory mask (tgt_len, src_len)
            src_key_padding_mask (Optional[Tensor]): source padding mask (batch_size, src_len)
            tgt_key_padding_mask (Optional[Tensor]): target padding mask (batch_size, tgt_len)
            memory_key_padding_mask (Optional[Tensor]): memory padding mask (batch_size, src_len)
        
        Output:
            Tensor: output sequence (batch_size, tgt_len, d_model)
        """
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output