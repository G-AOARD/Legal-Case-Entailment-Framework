from collections import defaultdict

import torch
import numpy as np
from scipy.special import softmax

from tqdm import tqdm

from src.llms.prompts.document_ranking import PROMPT_DICT as DR_PROMPT_DICT
from src.llms.prompts.feature_extraction import PROMPT_DICT as FE_PROMPT_DICT
from src.utils.model_utils import extract_embeddings


def predict_seq_cls(
    test_dataset,
    model,
    tokenizer,
    batch_size=8,
    max_length=512,
    prompt=None,
):
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size)
    if prompt is not None:
        query_input_text = DR_PROMPT_DICT[prompt]["query"]
        doc_input_text = DR_PROMPT_DICT[prompt]["doc"]
    else:
        query_input_text = doc_input_text = "{text}"

    model.eval()
    scores = defaultdict(lambda: {})
    for batch in tqdm(test_dataloader):
        inputs = tokenizer(
            [query_input_text.format_map({"text": text}) for text in batch[0]],
            [doc_input_text.format_map({"text": text}) for text in batch[1]],
            padding=True,
            max_length=max_length,
            truncation="longest_first",
            return_tensors='pt'
        )
        for key in inputs.keys():
            inputs[key] = inputs[key].to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()
        if logits.shape[-1] == 2:
            probs = softmax(logits, axis=-1) * 100
        else:
            probs = np.exp(logits) * 100
        score = [prob[-1] for prob in probs]
        for i, (query_id, doc_id) in enumerate(
                zip(batch[3]["query_id"], batch[3]["doc_id"])):
            scores[query_id][doc_id] = score[i].item()
    return scores


def predict_feature_distance(
    test_dataset,
    model,
    tokenizer,
    use_remote_encode=False,
    batch_size=8,
    max_length=512,
    prompt=None,
    embeddings_method=None,
    add_eos_token=False,
    normalize=False,
):
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    if prompt is not None:
        query_input_template = FE_PROMPT_DICT[prompt]["query"]
        doc_input_template = FE_PROMPT_DICT[prompt]["doc"]
    else:
        query_input_template = doc_input_template = "{text}"

    scores = defaultdict(lambda: defaultdict(lambda: 0))
    for batch in tqdm(test_dataloader):
        query_texts = batch[0]
        doc_texts = batch[1]

        query_embeddings = extract_embeddings(
            model, tokenizer, query_texts,
            use_remote_encode=use_remote_encode,
            batch_size=batch_size,
            max_length=max_length,
            input_template=query_input_template,
            add_eos_token=add_eos_token,
            embeddings_method=embeddings_method,
            normalize=normalize
        )
        doc_embeddings = extract_embeddings(
            model, tokenizer, doc_texts,
            use_remote_encode=use_remote_encode,
            batch_size=batch_size,
            max_length=max_length,
            input_template=doc_input_template,
            add_eos_token=add_eos_token,
            embeddings_method=embeddings_method,
            normalize=normalize
        )

        if normalize:
            multiply_factor = 100
        else:
            multiply_factor = 1
        score = (query_embeddings * doc_embeddings).sum(-1) * multiply_factor
        for i, (query_id, doc_id) in enumerate(
                zip(batch[3]["query_id"], batch[3]["doc_id"])):
            scores[query_id][doc_id] = max(scores[query_id][doc_id], score[i].item())
    return scores
