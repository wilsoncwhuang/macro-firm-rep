import torch
import torch.nn as nn
from models import MLP, Transformer, FIM


class MacroAttention(nn.Module):
    def __init__(self, d_enc=1536, d_model=256, num_heads=4, dropout=0.1, T_max=12):
        super().__init__()
        self.d_model = d_model
        
        # projections
        self.proj_q = nn.Linear(d_enc, d_model)
        self.proj_kv = nn.Linear(d_enc, d_model)
        self.mhattn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True, bias=False)

        # learned positional embeddings
        self.macro_pos = nn.Embedding(T_max, d_model)

    def forward(self, company_embedding, macro_embedding, macro_mask=None):
        B, T, C, D = macro_embedding.shape 
        S = T * C

        # projection
        q = self.proj_q(company_embedding) # (B, 1, 256)
        kv = self.proj_kv(macro_embedding.view(B, S, -1))  # (B, S, 256)

        # macro position encoding
        macro_ids = torch.arange(T, device=macro_embedding.device)
        macro_pos_enc = self.macro_pos(macro_ids) 
        macro_pos_enc = macro_pos_enc.unsqueeze(1).expand(T, C, self.d_model)
        kv = kv + macro_pos_enc.reshape(1, S, self.d_model)  # (B, S, 256)

        # key padding mask
        kp_mask = ~macro_mask.reshape(B, S)

        # handle all-pad rows in key padding mask
        all_pad = kp_mask.all(dim=1)
        if all_pad.any():
            all_pad_idx = all_pad.nonzero(as_tuple=True)[0]
            kp_mask[all_pad_idx, 0] = False
            kv[all_pad_idx, 0].zero_()
        
        attn_output, attn_weights = self.mhattn(q, kv, kv, key_padding_mask=kp_mask)  # (B, 1, 256) (B, 1, S)
        if all_pad.any():
            attn_weights[all_pad] = 0.0
        
        return attn_output, attn_weights


class DefaultFiLM(nn.Module):
    def __init__(self, feature_size=16, d_model=256, d_bottleneck=32, dropout=0.1, T_max=12):
        super().__init__()
        
        self.proj_macro = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_bottleneck),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # FiLM parameters
        self.gamma = nn.Linear(d_bottleneck, feature_size) # (B, 16)
        self.beta = nn.Linear(d_bottleneck, feature_size) # (B, 16)

        self.ln = nn.LayerNorm(feature_size, elementwise_affine=False)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

        # learned positional embeddings        
        self.x_pos = nn.Embedding(T_max, feature_size)

    def forward(self, x, m):
        B, T, D = x.shape

        # company feature position encoding
        x_ids = torch.arange(T, device=x.device)
        x_pos_enc = self.x_pos(x_ids)
        x = x + x_pos_enc.unsqueeze(0) 

        # FiLM
        m = self.proj_macro(m.squeeze(1))
        gamma = torch.tanh(self.gamma(m))
        beta = self.beta(m)
        return self.ln(gamma.unsqueeze(1) * x + beta.unsqueeze(1)) # (B, 12, 16)



class model(nn.Module):
    def __init__(self, feature_size, window_size, horizon_size, backbone, d_enc=1536, d_model=256, d_bottleneck=32, dropout=0):
        super().__init__()

        # cross attention + FiLM
        self.macro = MacroAttention(d_enc, d_model)
        self.film = DefaultFiLM(feature_size, d_model, d_bottleneck)
        self.backbone = backbone
        
        if self.backbone == "mlp":
            print("Backbone: MLP")
            self.net = MLP.model(
                feature_size=feature_size,
                window_size=window_size,
                horizon_size=horizon_size,
                dropout=dropout,
                layer=True
            )
        elif self.backbone == "transformer":
            print("Backbone: Transformer")
            self.net = Transformer.model(
                enc_in=feature_size,
                d_model=64,
                e_layers=3,
                n_heads=4,
                freq="m",
                output_attention=True,
                factor=1,
                d_ff=512,
                activation="gelu",
                embed="timeF",
                seq_len=window_size, # special argument from Time Series Library (can change)
                num_class=horizon_size,
                horizon_size=horizon_size,
                dropout=dropout,
                layer=True
            )
        elif self.backbone == "fim":
            print("Backbone: FIM")
            self.net = FIM.model(
                feature_size=feature_size,
                window_size=window_size,
                horizon_size=horizon_size,
                layer=True
            )

    def forward(self, x, company_embeddings, macro_embeddings, macro_mask):
        
        macro_context, attn_weights = self.macro(company_embeddings, macro_embeddings, macro_mask)
        x = self.film(x, macro_context) 
         
        result = self.net(x)
        return result