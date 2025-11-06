"""
encoders.py

PyTorch modules for time-series encoding and graph encoding.
- TimeSeriesEncoderLSTM: per-stock LSTM encoder producing d_time vector
- TimeSeriesEncoderTransformer: optional transformer encoder
- GraphSAGE-like aggregator: simple aggregate + MLP
- CombinedModel: convenience wrapper to go from (N,W,F) to (N,d_gnn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# --------------------------
# Time-series encoders
# --------------------------
class TimeSeriesEncoderLSTM(nn.Module):
    def __init__(self, input_dim: int, d_time: int = 64, num_layers: int = 2, bidirectional: bool = False, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, d_time, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.d_time = d_time * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, W, F) or (N, W, F)
        returns: (B, N, d_time) or (N, d_time)
        """
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0); single = True
        B, N, W, F = x.shape
        x = x.view(B * N, W, F)  # (B*N, W, F)
        out, (h, c) = self.lstm(x)  # out: (B*N, W, d_time)
        last = out[:, -1, :]        # (B*N, d_time)
        last = last.view(B, N, -1)
        if single:
            return last.squeeze(0)
        return last

class TimeSeriesEncoderTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, W, F) or (N, W, F)
        returns: (B, N, d_model) or (N, d_model)
        """
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0); single = True
        B, N, W, F = x.shape
        x = x.view(B * N, W, F)
        z = self.input_fc(x)
        z = self.transformer(z)  # (B*N, W, d_model)
        z = z.mean(dim=1)
        z = z.view(B, N, -1)
        if single:
            return z.squeeze(0)
        return z

# --------------------------
# Graph aggregator (GraphSAGE-like)
# --------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, activation=F.relu):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = in_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_d * 2, hidden_dim))
        self.activation = activation
        self.out_dim = hidden_dim

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        H: (B, N, d) or (N, d)
        A: (N, N) or (B, N, N)
        returns H_out (same batch behavior) shape (B, N, out_dim) or (N, out_dim)
        """
        single = False
        if H.dim() == 2:
            H = H.unsqueeze(0); single = True
        B, N, d = H.shape
        if A.dim() == 2:
            A_batch = A.unsqueeze(0).repeat(B, 1, 1)
        else:
            A_batch = A
        h = H
        for lin in self.layers:
            neigh = torch.bmm(A_batch, h)  # (B,N,d)
            cat = torch.cat([h, neigh], dim=2)  # (B,N,2d)
            h = self.activation(lin(cat))
        if single:
            return h.squeeze(0)
        return h

# --------------------------
# Combined wrapper
# --------------------------
class CombinedEncoder(nn.Module):
    def __init__(self, input_dim: int, W: int = 180, d_time: int = 64, d_gnn: int = 64, time_layers: int = 2, gnn_layers: int = 2, use_transformer: bool = False):
        super().__init__()
        self.use_transformer = use_transformer
        if use_transformer:
            self.time_enc = TimeSeriesEncoderTransformer(input_dim, d_model=d_time, num_layers=time_layers)
            self.time_out_dim = d_time
        else:
            self.time_enc = TimeSeriesEncoderLSTM(input_dim, d_time=d_time, num_layers=time_layers)
            self.time_out_dim = self.time_enc.d_time
        self.gnn = GraphSAGE(self.time_out_dim, hidden_dim=d_gnn, num_layers=gnn_layers)
        self.d_gnn = d_gnn

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        X: (B, N, W, F) or (N, W, F)
        A: (N, N) or (B, N, N)
        returns: H_gnn (B, N, d_gnn) or (N, d_gnn)
        """
        H_time = self.time_enc(X)  # (B, N, d_time)
        H_gnn = self.gnn(H_time, A)  # (B, N, d_gnn)
        return H_gnn
