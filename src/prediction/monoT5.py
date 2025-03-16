import os
import sys
import fire
import itertools
from collections import defaultdict

import numpy as np
from tqdm import tqdm

sys.path.append("modules/pygaggle")
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5

from src.coliee_dataset import LegalCaseEntailmentDataset
from src.training.metrics import F1Score, MRR
from src.utils.model_utils import load_checkpoint_tokenizer_configs


def predict_monoT5(test_dataset, model):
    model.eval()
    reranker = MonoT5(model=model)

    query_scores = defaultdict(lambda: {})
    for query_batch in tqdm(test_dataset, desc="MonoT5 prediction"):
        query_id = query_batch[0][3]["query_id"]
        query = Query(query_batch[0][0])

        doc_ids = [query_batch[i][3]["doc_id"] for i in range(len(query_batch))]
        doc_texts = [query_batch[i][1] for i in range(len(query_batch))]
        docs = [
            Text(text, metadata={"doc_id": text_id})
            for text_id, text in zip(doc_ids, doc_texts)
        ]

        score = defaultdict(lambda: 0)
        result = reranker.rerank(query, docs)
        for c in result:
            score[c.metadata["doc_id"]] = max(
                score[c.metadata["doc_id"]], np.exp(c.score) * 100)
        query_scores[query_id] = score

    return query_scores


def evaluate_monoT5(
    test_dataset=None,
    dataset_path=None,
    dataset_name=None,
    eval_segment="test",
    model=None,
    checkpoint_path=None,
    query_retrieval_dict=None,
    top_k=2,
    margin=1,
    threshold=90,
    **kwargs
):
    if test_dataset is None:
        test_dataset = LegalCaseEntailmentDataset(
            dataset_path,
            dataset_name,
            model_type="T5",
            segment=eval_segment,
            query_retrieval_dict=query_retrieval_dict
        )
    labels = test_dataset.labels

    if model is None:
        model, _, _ = load_checkpoint_tokenizer_configs(
            checkpoint_path,
            model_type="T5",
            **kwargs
        )
        model = model.to("cuda")
    query_scores = predict_monoT5(test_dataset, model)

    if not isinstance(top_k, list):
        top_k = [top_k]
    if not isinstance(margin, list):
        margin = [margin]
    if not isinstance(threshold, list):
        threshold = [threshold]

    grid_configs = list(itertools.product(*[top_k, margin, threshold]))

    metrics = [F1Score(), MRR()]
    results = {"best": None}
    best_metrics, best_config = {}, {}
    predictions = {}
    for query, scores in query_scores.items():
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        sorted_scores = {k + ".txt": v for k, v in sorted_scores}
        predictions[query] = sorted_scores

    for ci, (k, m, thresh) in enumerate(grid_configs):
        for query, scores in predictions.items():
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
            pred = [sorted_scores[0][0]]
            for doc, score in sorted_scores[1:]:
                if scores[pred[0]] - score < m and score >= thresh:
                    pred.append(doc)

            for i in range(len(metrics)):
                metrics[i].add(reference=labels[query], prediction=pred)

        conf_results = {}
        for i in range(len(metrics)): 
            conf_results = {**conf_results, **metrics[i].compute()}
        results[f"k={k}_m={m}_thresh={thresh}"] = conf_results
        if conf_results["f1"] > best_metrics.get("f1", -1):
            best_metrics = conf_results
            best_config = [k, m, thresh]

    results["best"] = {
        "metrics": best_metrics,
        "config": best_config
    }
    return results


if __name__ == "__main__":
    fire.Fire(evaluate_monoT5)