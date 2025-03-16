import torch

from transformers import (
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding
)

from src.llms.prompts.instruction_finetune import (
    PROMPT_DICT,
    INSTRUCTION_DICT
)


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
            self,
            data_source,
            batch_size, 
            drop_last=False,
            shuffle=True):
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths)
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i: i + self.batch_size]
                   for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


def get_dataloader(
    dataset,
    configs,
    tokenizer,
    is_train=False
):
    if configs.get("sorted_by_length", False):
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=configs.get("num_workers_dataloader", 1),
            pin_memory=True,
            batch_sampler=LengthBasedBatchSampler(
                dataset,
                batch_size=configs["batch_size"] if is_train else configs["eval_batch_size"],
                drop_last=False,
                shuffle=is_train
            ),
            collate_fn=get_collator(configs, tokenizer)
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=configs.get("num_workers_dataloader", 1),
            pin_memory=True,
            batch_size=configs["batch_size"] if is_train else configs["eval_batch_size"],
            shuffle=is_train,
            collate_fn=get_collator(configs, tokenizer),
        )


def get_collator(
    configs,
    tokenizer
):
    if configs["model_type"] in ["CAUSAL_LM"]:
        if configs.get("reranking_batching", False):
            return InstructionBatchCollator(
                tokenizer,
                max_length=configs["context_length"],
                mask_inputs=configs["mask_inputs"],
                prompt_template=configs["prompt_template"],
                instruction_template=configs["instruction_template"]
            )
        else:
            return DataCollatorForSeq2Seq(tokenizer)
    elif configs["model_type"] in ["FEATURE_EXTRACTION"]:
        return DataCollatorWithPadding(tokenizer)
    elif configs["model_type"] in ["SEQ_CLS"]:
        if configs.get("reranking_batching", False):
            return SeqClsRerankingBatchCollator(
                tokenizer,
                max_length=configs["context_length"],
                prompt_template=configs["prompt_template"],
            )
        else:
            return DataCollatorWithPadding(tokenizer)
    elif configs["model_type"] in ["T5"]:
        return Seq2SeqRerankingBatchCollator(
            tokenizer,
            max_length=configs["context_length"],
            prompt_template=configs["prompt_template"],
            instruction_template=configs["instruction_template"]
        )
    else:
        raise ValueError(configs["model_type"])


class InstructionBatchCollator:
    def __init__(
        self,
        tokenizer,
        max_length=None,
        mask_inputs=True,
        ignore_index=-100,
        prompt_template="alpaca",
        instruction_template="relevancy"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_inputs = mask_inputs
        self.ignore_index = ignore_index

        self.instruction_text = PROMPT_DICT[prompt_template]["instruction"].format_map(
            {"instruction": INSTRUCTION_DICT[instruction_template]["instruction"]}
        )
        self.instruction_input_text = PROMPT_DICT[prompt_template]["input"]
        self.encoded_response_text = self.tokenizer.encode(PROMPT_DICT[prompt_template]["response"])
        self.input_template = INSTRUCTION_DICT[instruction_template]["input"]
        self.answer_list = INSTRUCTION_DICT[instruction_template]["answers"]

        self.input_max_length = max_length - len(self.encoded_response_text)

    def __call__(self, batch):
        input_texts = [
            self.input_template.format_map({"query": example[0], "doc": example[1]})
            for b in batch for example in b
        ]
        answer_texts = [self.answer_list[example[2]] for b in batch for example in b]

        instruction_texts = [
            self.instruction_text + self.instruction_input_text.format_map({"input": text})
            for text in input_texts
        ]
        answer_texts = [self.answer_list[example[2]] for b in batch for example in b]

        encoded_inputs = self.tokenizer.batch_encode_plus(
            instruction_texts, max_length=self.input_max_length + 1)
        encoded_inputs["input_ids"] = [
            encoded_input[:-1] + self.encoded_response_text
            for encoded_input in encoded_inputs["input_ids"]
        ]
        encoded_inputs["attention_mask"] = [
            mask[:-1] + [1] * len(self.encoded_response_text)
            for mask in encoded_inputs["attention_mask"]
        ]
        encoded_inputs = self.tokenizer.pad(encoded_inputs, padding="max_length")
        encoded_instructions = torch.tensor(encoded_inputs["input_ids"], dtype=torch.int64)
        attention_mask = torch.tensor(
            encoded_inputs["attention_mask"], dtype=torch.int64)
        encoded_answers = self.tokenizer(answer_texts, return_tensors="pt")["input_ids"]
        return {
            "input_ids": encoded_instructions,
            "attention_mask": attention_mask,
            "labels": encoded_answers
        }


class Seq2SeqRerankingBatchCollator:
    def __init__(
        self,
        tokenizer,
        max_length=None,
        prompt_template="monoT5",
        instruction_template="monoT5",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.instruction_text = PROMPT_DICT[prompt_template]["instruction"].format_map(
            {"instruction": INSTRUCTION_DICT[instruction_template]["instruction"]}
        )
        self.instruction_input_text = PROMPT_DICT[prompt_template]["input"]
        self.encoded_response_text = self.tokenizer.encode(PROMPT_DICT[prompt_template]["response"])
        self.input_template = INSTRUCTION_DICT[instruction_template]["input"]
        self.answer_list = INSTRUCTION_DICT[instruction_template]["answers"]

        self.input_max_length = max_length - len(self.encoded_response_text)

    def __call__(self, batch):
        input_texts = [
            self.input_template.format_map({"query": example[0], "doc": example[1]})
            for b in batch for example in b
        ]
        answer_texts = [self.answer_list[example[2]] for b in batch for example in b]

        instruction_texts = [
            self.instruction_text + self.instruction_input_text.format_map({"input": text})
            for text in input_texts
        ]
        answer_texts = [self.answer_list[example[2]] for b in batch for example in b]

        encoded_inputs = self.tokenizer.batch_encode_plus(
            instruction_texts, max_length=self.input_max_length + 1)
        encoded_inputs["input_ids"] = [
            encoded_input[:-1] + self.encoded_response_text
            for encoded_input in encoded_inputs["input_ids"]
        ]
        encoded_inputs["attention_mask"] = [
            mask[:-1] + [1] * len(self.encoded_response_text)
            for mask in encoded_inputs["attention_mask"]
        ]
        encoded_inputs = self.tokenizer.pad(encoded_inputs, padding="max_length")
        encoded_instructions = torch.tensor(encoded_inputs["input_ids"], dtype=torch.int64)
        attention_mask = torch.tensor(
            encoded_inputs["attention_mask"], dtype=torch.int64)
        encoded_answers = self.tokenizer(answer_texts, return_tensors="pt")["input_ids"]
        return {
            "input_ids": encoded_instructions,
            "attention_mask": attention_mask,
            "labels": encoded_answers
        }


class SeqClsRerankingBatchCollator:
    def __init__(
        self,
        tokenizer,
        max_length=None,
        prompt_template="rankllama",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template

    def __call__(self, batch):
        query_texts = [
            PROMPT_DICT[self.prompt_template]["query"].format_map({"query": example[0]})
            for b in batch for example in b
        ]
        doc_texts = [
            PROMPT_DICT[self.prompt_template]["doc"].format_map({"doc": example[1]})
            for b in batch for example in b
        ]
        inputs = self.tokenizer(
            query_texts,
            doc_texts,
            max_length=self.max_length,
            padding=True,
            truncation="longest_first",
            return_tensors="pt"
        )
        inputs["labels"] = torch.tensor([example[2] for b in batch for example in b], dtype=torch.int64)
        return inputs