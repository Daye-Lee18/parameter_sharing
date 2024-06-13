import torch
import torch.nn as nn

# https://github.com/takase/share_layer_params/blob/main/fairseq/fairseq/models/transformer.py#L275
# TransformerEncoder(), ,,, code 참고 
import torch
import torch.nn as nn

class Encoder(nn.TransformerEncoder):
    def __init__(
        self,
        input_dim,
        d_model=512,
        nhead=16,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        num_unique_layers=3,
        num_total_layers=6,
        mode="cycle_rev",
        norm=False,
        max_token=1000,
    ):
        assert mode in {"sequence", "cycle", "cycle_rev"}
        quotient, remainder = divmod(num_total_layers, num_unique_layers)
        assert remainder == 0 

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, batch_first=True
        )
        super().__init__(encoder_layer, num_layers=num_unique_layers, norm=norm)
        self.N = num_total_layers
        self.M = num_unique_layers
        self.mode = mode
        self.norm = nn.LayerNorm(d_model) if norm else None
        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_token, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(torch.device("cpu"))

        print(f"Encoder structure: {self.layers}")
        
    def forward(self, x, mask=None, src_key_padding_mask=None, verbose=False):
        """
        Input: 
            - x (Tensor): input sequence tensor (batch_size, seq_len)
            - mask (Optional[Tensor]): future token masking (seq_len, seq_len)
            - src_key_padding_mask (Optional[Tensor]): indicating the padding token's positions with True value (batch_size, seq_len)
            - verbose (bool): for debugging
        
        Output:
            x (Tensor): encoded output (batch_size, seq_len, d_model)
        """
        # Debugging print to check the shape and dtype of x
        # print(f"Initial x shape: {x.shape}, dtype: {x.dtype}")
        # print(f"Initial x max value: {x.max()}, x min value: {x.min()}")

        # Ensure x is of type Long
        x = x.long()

        # # Check for NaN or Inf values
        # if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
        #     raise ValueError("Input x contains NaN or Inf values")

        # # Ensure x values are within the range [0, input_dim-1]
        # if torch.any(x >= self.tok_embedding.num_embeddings) or torch.any(x < 0):
        #     print(x)
        #     raise ValueError("Input x contains values out of range for tok_embedding")

        # # Debugging print to check the shape and dtype of x after conversion
        # print(f"Converted x shape: {x.shape}, dtype: {x.dtype}")
        # print(f"Converted x max value: {x.max()}, x min value: {x.min()}")

        # # Additional debug print before embedding
        # try:
        #     embedded_x = self.tok_embedding(x)
        #     print(f"embedded_x shape: {embedded_x.shape}, dtype: {embedded_x.dtype}")
        # except Exception as e:
        #     print(f"Embedding error: {e}")
        #     print(f"x max value: {x.max()}, x min value: {x.min()}")
        #     raise

        # Save original x
        original_x = x.clone()

        x_pos = torch.arange(0, original_x.size(1)).unsqueeze(0).repeat(original_x.size(0), 1).to(original_x.device)
        self.scale = self.scale.to(original_x.device)

        x = self.dropout((self.tok_embedding(original_x) * self.scale) + self.pos_embedding(x_pos))
        
        cnt = 0 
        for enc_i in range(self.N):
            if enc_i == 0:
                enc_i = cnt
            elif self.mode == "sequence":
                if (enc_i) % math.floor(self.N/self.M) == 0:
                    cnt += 1 
                    enc_i = cnt
                else:
                    enc_i = cnt
            elif self.mode == "cycle":
                if enc_i < self.M: 
                    cnt += 1
                    enc_i = cnt
                else:
                    enc_i = ((enc_i) % self.M )
            elif self.mode == "cycle_rev":
                if enc_i < self.M:
                    cnt += 1 
                    enc_i = cnt
                elif (enc_i) < self.M * (round(self.N/self.M,0)-1):
                    enc_i = ((enc_i)% self.M)
                else:
                    enc_i = self.M - ((enc_i)%self.M) -1
            else:
                assert self.mode == "sequence" or self.mode == "cycle" or self.mode == "cycle_rev"
            if verbose:
                print(f"layer {enc_i}")

            x = self.layers[enc_i](x)
            

        if self.norm is not None:
            x = self.norm(x)
        
        x = x.transpose(0, 1)  # Change back to (batch_size, seq_len, d_model)
        return x

class Decoder(nn.TransformerDecoder):
    def __init__(
        self,
        output_dim,
        d_model=512,
        nhead=16,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        num_unique_layers=3,
        num_total_layers=6,
        mode="cycle_rev",
        norm=False,
        max_token=1000,
    ):
        assert mode in {"sequence", "cycle", "cycle_rev"}
        quotient, remainder = divmod(num_total_layers, num_unique_layers)
        assert remainder == 0


        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, batch_first=True
        )
        
        super().__init__(decoder_layer, num_layers=num_unique_layers, norm=norm)
        self.N = num_total_layers
        self.M = num_unique_layers
        self.mode = mode
        self.norm = nn.LayerNorm(d_model) if norm else None
        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_embedding = nn.Embedding(max_token, d_model)
        self.fc_out1 = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(torch.device("cpu"))

        print(f"Decoder structure: {self.layers}")

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, verbose=False):
        # Ensure tgt is of type Long
        tgt = tgt.long()

        # # Check for NaN or Inf values
        # if torch.any(torch.isnan(tgt)) or torch.any(torch.isinf(tgt)):
        #     raise ValueError("Input tgt contains NaN or Inf values")

        # # Ensure tgt values are within the range [0, output_dim-1]
        # if torch.any(tgt >= self.tok_embedding.num_embeddings) or torch.any(tgt < 0):
        #     print(tgt)
        #     raise ValueError("Input tgt contains values out of range for tok_embedding")
        
        # # Debugging print to check the shape and dtype of tgt
        # print(f"tgt shape: {tgt.shape}, dtype: {tgt.dtype}")
        # print(f"tgt max value: {tgt.max()}, tgt min value: {tgt.min()}")

        # # Additional debug print before embedding
        # try:
        #     embedded_tgt = self.tok_embedding(tgt)
        #     print(f"embedded_tgt shape: {embedded_tgt.shape}, dtype: {embedded_tgt.dtype}")
        # except Exception as e:
        #     print(f"Embedding error: {e}")
        #     print(f"tgt max value: {tgt.max()}, tgt min value: {tgt.min()}")
        #     raise

        tgt_pos = torch.arange(0, tgt.size(1)).unsqueeze(0).repeat(tgt.size(0), 1).to(tgt.device)
        self.scale = self.scale.to(tgt.device)

        tgt = self.dropout((self.tok_embedding(tgt) * self.scale) + self.pos_embedding(tgt_pos))
        
        cnt = 0 
        for dec_i in range(self.N):
            if dec_i == 0:
                dec_i = cnt
            elif self.mode == "sequence":
                if (dec_i) % math.floor(self.N/self.M) == 0:
                    cnt += 1 
                    dec_i = cnt
                else:
                    dec_i = cnt
            elif self.mode == "cycle":
                if dec_i < self.M: 
                    cnt += 1
                    dec_i = cnt
                else:
                    dec_i = ((dec_i) % self.M )
            elif self.mode == "cycle_rev":
                if dec_i < self.M:
                    cnt += 1 
                    dec_i = cnt
                elif (dec_i) < self.M * (round(self.N/self.M,0)-1):
                    dec_i = ((dec_i)% self.M)
                else:
                    dec_i = self.M - ((dec_i)%self.M) -1
            else:
                assert self.mode == "sequence" or self.mode == "cycle" or self.mode == "cycle_rev"
            if verbose:
                print(f"layer {dec_i}")

            tgt = self.layers[dec_i](tgt, memory, tgt_mask, memory_mask,
                                     tgt_key_padding_mask, memory_key_padding_mask)
        if self.norm is not None:
            tgt = self.norm(tgt)

        tgt = tgt.transpose(0, 1)  # Change back to (batch_size, tgt_len, d_model)
        output = self.fc_out1(tgt)
      
        return output

class ParameterShareTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, src_pad_idx, tgt_pad_idx, max_token, device, mode, num_unique_layers,  num_total_layers, 
                d_model=256, nhead=16, dim_feedforward=1024, dropout=0.1, activation="relu", norm=False):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, nhead, dim_feedforward, dropout, activation, 
                               num_unique_layers, num_total_layers, mode, norm, max_token)
        self.decoder = Decoder(output_dim, d_model, nhead, dim_feedforward, dropout, activation, 
                               num_unique_layers, num_total_layers, mode, norm, max_token)
        self._reset_parameters()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = tgt_pad_idx
        self.device = device

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # print(f"src_shape in Transformer: {src.shape}") # (bs, seq_len)
        # print(f"tgt_shape in Transformer: {tgt.shape}") # (bs, seq_len)
        # print(f"src_mask shape in Transformer: {src_mask.shape}") # (seq_len, seq_len)
        # print(f"src_padding_mask shape in Transformer: {src_key_padding_mask.shape}") # (bs, seq_len)
        # print(f"tgt_mask shape in Transformer: {tgt_mask.shape}") # (seq_len, seq_len)
        # print(f"memory_mask shape in Transformer: {memory_mask.shape}") # (tgt_seq_len, bs, src_seq_len)
        """
        In the PyTorch language, 
        the original Transformer settings are src_mask=None and memory_mask=None, 
        and for tgt_mask=generate_square_subsequent_mask(T).
        Again, memory_mask is used only when you don’t want to let the decoder attend certain tokens in the input sequence
        """
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask, verbose=False) # (seq_len, bs, d_model)
        # print(f"encoder output memory shape in Transformer: {memory.shape}") 
        
        # teacher forcing: use the original tgt data, allow the model to predict the "next" token  
        # batch_first=True (bs, seq_len, d_model) needed 
        
        output = self.decoder(tgt, memory.transpose(0,1), verbose=False)
        # output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
        #                       tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        return output

