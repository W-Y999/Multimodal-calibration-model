import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    """共享文本编码器（Embedding + CNN + GRU）"""
    
    def __init__(self):
        super().__init__()
        # TODO: define embedding layer
        # TODO: define TextCNN layers
        # TODO: define BiGRU
        pass

    def forward(self, x):
        """
        x: tokenized input (batch, seq_len)
        return: text vector (batch, hidden_dim)
        """
        # TODO: implement forward pass
        pass


class NumericEncoder(nn.Module):
    """数值特征编码器 MLP"""

    def __init__(self):
        super().__init__()
        # TODO: define MLP layers
        pass

    def forward(self, x):
        """
        x: numeric features tensor
        return: (batch, hidden_dim)
        """
        # TODO: implement forward pass
        pass


class MultiModalRegressor(nn.Module):
    """多模态融合 + 回归头"""

    def __init__(self):
        super().__init__()
        # TODO: instantiate TextEncoder
        # TODO: instantiate NumericEncoder
        # TODO: define fusion MLP for regression
        pass

    def forward(self, text_answer, text_rule, numeric_feats):
        """
        text_answer: token IDs
        text_rule: token IDs
        numeric_feats: numeric input
        return: score (batch, 1)
        """
        # TODO: call submodules and fuse outputs
        pass
