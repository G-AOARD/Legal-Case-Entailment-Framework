import numpy as np

import torch
from torch.utils.data import Dataset

from src.llms.prompts.instruction_finetune import PROMPT_DICT, INSTRUCTION_DICT


def format_prompt(prompt_template, instruction_template, instruction_args):
    instruction_str = PROMPT_DICT[prompt_template]["instruction"].format_map({
        "instruction": INSTRUCTION_DICT[instruction_template]["instruction"]
    })
    input_str = PROMPT_DICT[prompt_template]["input"].format_map({
        "input": INSTRUCTION_DICT[instruction_template]["input"].format_map(instruction_args)
    })
    begin_answer_str = PROMPT_DICT[prompt_template]["response"]
    return instruction_str, input_str, begin_answer_str


class InstructionDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length=None,
        mask_inputs=True,
        ignore_index=-100,
        prompt_template="alpaca",
        instruction_template="relevancy"
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length or 4096
        self.mask_inputs = mask_inputs
        self.ignore_index = ignore_index
        self.prompt_template = prompt_template
        self.instruction_template = instruction_template

    def build_input_dataset(self, *args, **kwargs):
        self.dataset.build_input_dataset(*args, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        instruction_str, input_str, begin_answer_str = format_prompt(
            self.prompt_template, self.instruction_template, instruction_args={
                "query": sample[0],
                "doc": sample[1],
            }
        )
        answer_str = INSTRUCTION_DICT[self.instruction_template]["answers"][sample[2]]

        encoded_instruction = self.tokenizer.encode(
            instruction_str + input_str, max_length=self.max_length)
        encoded_begin_answer = self.tokenizer.encode(begin_answer_str)[1:]
        encoded_ans = self.tokenizer.encode(answer_str)[1:] + [self.tokenizer.eos_token_id]

        max_input_len = self.max_length - len(encoded_begin_answer) - len(encoded_ans)
        encoded_prompt = encoded_instruction[:min(len(encoded_instruction), max_input_len)] + \
            encoded_begin_answer
        encoded_prompt_ans = encoded_prompt + encoded_ans

        encoded_prompt_tensor = torch.tensor(encoded_prompt_ans, dtype=torch.int64)
        attention_mask = encoded_prompt_tensor.ge(0)
        labels = encoded_prompt_tensor.clone()
        if self.mask_inputs:
            labels[:len(encoded_prompt)] = -1
        label_mask = labels.ge(0)
        encoded_prompt_tensor[~attention_mask] = 0
        labels[~label_mask] = self.ignore_index

        return {
            "input_ids": encoded_prompt_tensor.tolist(),
            "labels": labels.tolist(),
            "attention_mask": attention_mask.tolist()
        }


class Sequence2SequenceDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length=None,
        prompt_template="alpaca",
        instruction_template="relevancy"
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length or 4096
        self.prompt_template = prompt_template
        self.instruction_template = instruction_template

    def build_input_dataset(self, *args, **kwargs):
        self.dataset.build_input_dataset(*args, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        instruction_str, input_str, begin_answer_str = format_prompt(
            self.prompt_template, self.instruction_template, instruction_args={
                "query": sample[0],
                "doc": sample[1],
            }
        )
        answer_str = INSTRUCTION_DICT[self.instruction_template]["answers"][sample[2]]

        encoded_instruction = self.tokenizer.encode(
            instruction_str + input_str, max_length=self.max_length)
        encoded_begin_answer = self.tokenizer.encode(begin_answer_str)
        encoded_ans = self.tokenizer.encode(answer_str)

        max_input_len = self.max_length - len(encoded_begin_answer) - len(encoded_ans)
        encoded_prompt = encoded_instruction[:min(len(encoded_instruction), max_input_len)] + \
            encoded_begin_answer
        encoded_prompt_ans = encoded_prompt + encoded_ans

        encoded_prompt_tensor = torch.tensor(encoded_prompt_ans, dtype=torch.int64)
        attention_mask = encoded_prompt_tensor.ge(0)
        encoded_prompt_tensor[~attention_mask] = 0

        return {
            "input_ids": encoded_prompt_tensor.tolist(),
            "labels": encoded_ans.tolist(),
            "attention_mask": attention_mask.tolist()
        }


class SequenceClassificationDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length=None,
        prompt_template="rankllama",
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length or 4096
        self.prompt_template = prompt_template

    def build_input_dataset(self, *args, **kwargs):
        self.dataset.build_input_dataset(*args, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        query_str = PROMPT_DICT[self.prompt_template]["query"].format_map({"query": sample[0]})
        doc_str = PROMPT_DICT[self.prompt_template]["doc"].format_map({"doc": sample[1]})
        inputs = self.tokenizer(
            query_str,
            doc_str,
            max_length=self.max_length,
            padding=True,
            truncation="longest_first"
        )
        inputs["labels"] = sample[2]
        return inputs
