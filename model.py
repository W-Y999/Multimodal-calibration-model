import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """共享文本编码器：Embedding + TextCNN + BiGRU"""

    def __init__(self, vocab_size=30000, embed_dim=128, cnn_filters=64, gru_hidden=128):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # TextCNN（3种卷积核）
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, cnn_filters, kernel_size=k)
            for k in [3, 4, 5]
        ])

        # BiGRU
        self.gru = nn.GRU(
            cnn_filters * 3, 
            gru_hidden, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        """
        x: token IDs → (batch, seq_len)
        """
        x = self.embedding(x)  # (B, L, E)
        x = x.transpose(1, 2)  # (B, E, L) for CNN

        cnn_outs = [F.relu(conv(x)) for conv in self.convs]  # list of (B, C, L')
        cnn_outs = [F.max_pool1d(c, c.shape[-1]).squeeze(-1) for c in cnn_outs]  # (B, C)
        cnn_out = torch.cat(cnn_outs, dim=1)  # (B, 3C)

        # GRU expects sequence format, so expand dim
        rnn_input = cnn_out.unsqueeze(1)  # (B, 1, 3C)
        _, h = self.gru(rnn_input)  # h: (2, B, H)
        h = torch.cat([h[0], h[1]], dim=1)  # (B, 2H)

        return h


class NumericEncoder(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class MultiModalRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_encoder = TextEncoder()
        self.numeric_encoder = NumericEncoder()

        fusion_dim = (128 * 2) + (128 * 2) + 64  # 拼接 answer + rule + numeric
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, text_answer, text_rule, numeric_feats):
        h_answer = self.text_encoder(text_answer)
        h_rule = self.text_encoder(text_rule)
        h_num = self.numeric_encoder(numeric_feats)

        fused = torch.cat([h_answer, h_rule, h_num], dim=1)
        pred = self.reg_head(fused)
        return pred
