import torch
import torch.nn as nn
from layers import FIM


class MacroProjectMean(nn.Module):
    def __init__(self, d_enc=1536, d_model=256):
        super().__init__()
        self.ln   = nn.LayerNorm(d_enc)
        self.proj = nn.Linear(d_enc, d_model)

    def forward(self, macro_embedding, macro_mask):
        # macro_embedding: (B,T,C,d_enc), macro_mask: (B,T,C) True=valid
        B, T, C, D = macro_embedding.shape
        m = self.ln(macro_embedding).reshape(B, T*C, D)          # (B,S,D)
        mask = macro_mask.reshape(B, T*C).float()                 # (B,S)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)     # (B,1)
        pooled = (m * mask.unsqueeze(-1)).sum(dim=1) / denom     # (B,D)
        context = self.proj(pooled).unsqueeze(1)                 # (B,1,256)
        return context, None


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
        self.beta = nn.Linear(d_bottleneck, feature_size) # B, 16)
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
    def __init__(self, feature_size, window_size, horizon_size, d_enc=1536, d_model=256, d_bottleneck=32, dropout=0):
        super().__init__()

        # projection + FiLM
        self.macro = MacroProjectMean(d_enc, d_model)
        self.film = DefaultFiLM(feature_size, d_model, d_bottleneck)
     
        # output mlp
        hidden_layer1 = nn.Linear(
            window_size * feature_size, window_size * feature_size
        )
        nn.init.xavier_uniform_(hidden_layer1.weight)
        hidden_layer2 = nn.Linear(
            window_size * feature_size, window_size * feature_size
        )
        nn.init.xavier_uniform_(hidden_layer2.weight)
        self.net = nn.Sequential(
            hidden_layer1,
            nn.BatchNorm1d(window_size * feature_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            hidden_layer2,
            nn.BatchNorm1d(window_size * feature_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            FIM(window_size * feature_size, horizon=horizon_size),
        )

    def forward(self, x, company_embeddings, macro_embeddings, macro_mask):
        
        macro_context, attn_weights = self.macro(macro_embeddings, macro_mask)
        x = self.film(x, macro_context) 

        x = x.view(x.shape[0], -1)
        result = self.net(x)
        return result