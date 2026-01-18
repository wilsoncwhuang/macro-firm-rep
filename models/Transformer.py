import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers import FIM


# This Transformer backbone implementation is adapted from:
# https://github.com/thuml/Time-Series-Library

class model(nn.Module):

    def __init__(self, enc_in, d_model, e_layers, n_heads, d_ff, seq_len, num_class,
                 output_attention=False, horizon_size=60, dropout=0.1, factor=3, activation='gelu', layer=False):
        super(model, self).__init__()

        self.layer = layer

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model * seq_len, num_class)

        self.x_pos = nn.Embedding(seq_len, enc_in)
        self.pos_project = nn.Linear(enc_in, d_model)
            
        self.fim = FIM(num_class, horizon=horizon_size)

    def forward(self, x_enc):

        B, T, D = x_enc.shape
        if self.layer == False:
            x_ids = torch.arange(T, device=x_enc.device)
            x_pos_enc = self.x_pos(x_ids)
            x_enc = x_enc + x_pos_enc.unsqueeze(0) 
            enc_out = self.pos_project(x_enc)
        else:
            enc_out = x_enc
            enc_out = self.pos_project(x_enc)
            
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        dec_out = self.fim(output)
        return dec_out  # [B, N]
