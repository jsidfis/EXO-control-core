import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim=14, output_dim=4, hidden_dim=128,
                 num_layers=2, dropout=0.3, bidirectional=True, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_dim = hidden_dim * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, fc_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim // 2, output_dim),
        )

    def forward(self, x, contact_mask=None):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class GRUBaseline(nn.Module):
    def __init__(self, input_dim=14, output_dim=4, hidden_dim=128,
                 num_layers=2, dropout=0.3, bidirectional=True, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_dim = hidden_dim * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, fc_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim // 2, output_dim),
        )

    def forward(self, x, contact_mask=None):
        gru_out, h_n = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        return self.fc(last_hidden)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBaseline(nn.Module):
    def __init__(self, input_dim=14, output_dim=4, d_model=128,
                 nhead=4, num_layers=4, dim_feedforward=256,
                 dropout=0.1, **kwargs):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, x, contact_mask=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        last_hidden = x[:, -1, :]
        return self.fc(last_hidden)
