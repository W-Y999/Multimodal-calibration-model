import torch
from model import MultiModalRegressor

def load_model(checkpoint_path):
    model = MultiModalRegressor()
    # TODO: load state dict
    return model

def predict(model, text_answer, text_rule, numeric_feats, tokenizer):
    """
    输入原始字符串 + 数值特征 → 输出预测合理性分数
    """
    # TODO: tokenize text_answer, text_rule
    # TODO: convert numeric_feats to tensor
    # TODO: call model
    pass


if __name__ == "__main__":
    # TODO: example usage
    pass

