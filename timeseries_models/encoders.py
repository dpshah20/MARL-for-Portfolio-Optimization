# models/encoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesEncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, bidirectional=False, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)
    def forward(self, x):
        """
        x: (B, N, W, F) or (N, W, F)
        Returns: (B, N, d_time) or (N, d_time) matching input
        """
        single = False
        if x.dim() == 3:  # (N,W,F)
            x = x.unsqueeze(0); single = True
        B, N, W, F = x.shape
        x = x.view(B*N, W, F)
        out, _ = self.lstm(x)
        h = out[:, -1, :]  # (B*N, d)
        h = h.view(B, N, -1)
        if single: return h.squeeze(0)
        return h

class TimeSeriesEncoderTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_dim = d_model
    def forward(self, x):
        single=False
        if x.dim()==3:
            x = x.unsqueeze(0); single=True
        B,N,W,F = x.shape
        x = x.view(B*N, W, F)
        z = self.input_fc(x)
        z = self.transformer(z)  # (B*N,W,d)
        h = z.mean(dim=1)
        h = h.view(B,N,-1)
        if single: return h.squeeze(0)
        return h

class GraphSAGELike(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = in_dim if i==0 else hidden_dim
            self.layers.append(nn.Linear(in_d*2, hidden_dim))
        self.activation = nn.ReLU()
        self.out_dim = hidden_dim
    def forward(self, H, A):
        """
        H: (B, N, d) or (N, d)
        A: (N,N) or (B,N,N)
        returns: same batch dim with shape (B,N,out_dim) or (N,out_dim)
        """
        single=False
        if H.dim()==2:
            H = H.unsqueeze(0); single=True
        B,N,d = H.shape
        h = H
        # ensure A has batch dim
        if A.ndim == 2:
            A_batch = A.unsqueeze(0).repeat(B,1,1)
        else:
            A_batch = A
        for lin in self.layers:
            neigh = torch.bmm(A_batch, h)  # (B,N,d)
            cat = torch.cat([h, neigh], dim=2)  # (B,N,2d)
            h = self.activation(lin(cat))
        if single:
            return h.squeeze(0)
        return h

class CombinedModel(nn.Module):
    def __init__(self, input_dim, W=30, d_time=64, d_gnn=64, lstm_layers=2):
        super().__init__()
        self.ts_enc = TimeSeriesEncoderLSTM(input_dim=input_dim, hidden_dim=d_time, num_layers=lstm_layers)
        self.gnn = GraphSAGELike(in_dim=self.ts_enc.out_dim, hidden_dim=d_gnn, num_layers=2)
    def forward(self, X, A):
        """
        X: (B, N, W, F) float tensor or (N, W, F)
        A: (N,N) or (B,N,N)
        returns: (B, N, d_gnn) or (N, d_gnn)
        """
        H_time = self.ts_enc(X)  # (B,N,d_time)
        H_gnn = self.gnn(H_time, A)  # (B,N,d_gnn)
        return H_gnn
