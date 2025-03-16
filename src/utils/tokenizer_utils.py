import os
import torch

from peft import PeftConfig
from transformers import AutoTokenizer


def load_tokenizer(
    model_name,
    max_length=None,
    peft_model=False,
):
    if peft_model:
        peft_config = PeftConfig.from_pretrained(model_name)
        model_name = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        max_length=max_length,
        trust_remote_code=True,
        token=os.getenv("HF_ACCESS_TOKEN")
    )
    if getattr(tokenizer, "pad_token_id", None) is None: 
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def tokens_to_words(tokens):
    words = []
    for i in range(len(tokens)):
        if tokens[i].startswith("##"):
            words[-1] = words[-1] + tokens[i][2:]
        else:
            words.append(tokens[i])
    return words