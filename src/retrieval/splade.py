import sys

import torch
from transformers import AutoTokenizer

sys.path.append("modules/splade")
from splade.models.transformer_rep import Splade


class SpladeHF:
    def __init__(self, model_path, device="cuda"):
        self.model = Splade(model_path, agg="max").to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            rep = self.model(d_kwargs=kwargs)["d_rep"]
        return rep
