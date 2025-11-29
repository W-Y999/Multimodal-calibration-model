import torch
from torch.utils.data import Dataset

class QADataDataset(Dataset):
    """读取用户作答文本、规则文本、数值特征、标签"""

    def __init__(self, json_path, tokenizer):
        """
        json_path: path to dataset.json
        tokenizer: any tokenizer (HuggingFace or custom)
        """
        # TODO: load JSON lines
        # TODO: store tokenizer
        pass

    def __len__(self):
        # TODO: return dataset length
        pass

    def __getitem__(self, idx):
        """
        返回：
        - tokenized text_answer
        - tokenized text_rule
        - numeric feature vector
        - label
        """
        # TODO: implement sample reading
        pass
