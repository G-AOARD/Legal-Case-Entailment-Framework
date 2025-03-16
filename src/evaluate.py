import os
import fire
import itertools
from collections import defaultdict

from pyserini.search import LuceneSearcher

from tqdm import tqdm

from src.coliee_dataset import get_segment_data, load_paragraph, LegalCaseEntailmentDataset
from src.prediction.monoT5 import predict_monoT5
from src.prediction.predict_utils import predict_feature_distance, predict_seq_cls
from src.retrieval.colbert_ret import ColBERT, predict_colbert
from src.retrieval.splade import SpladeHF
from src.utils.model_utils import load_checkpoint_tokenizer_configs
from src.training.metrics import F1Score
from src.utils import load_json, save_json, get_filename, sort_dict, aggregate_op


def run_retrieval(
    dataset_path,
    dataset_name,
    eval_segment,
    **retriever_configs
):
    if retriever_configs["model_name"] == "bm25":
        if not retriever_configs.get("indexes_dir", None):
            retriever_configs["indexes_dir"] = os.path.join(
                dataset_path, "bm25_indexes", dataset_name, eval_segment)
        retrieval_results = retrieval_bm25(
            dataset_path, dataset_name, eval_segment, **retriever_configs)
    else:
        test_dataset = LegalCaseEntailmentDataset(
            dataset_path, dataset_name, segment=eval_segment,
            model_type=retriever_configs["model_type"]
        )

        if retriever_configs["model_name"] in ["colbert", "colbert-uot"]:
            model = ColBERT(
                retriever_configs["model_path"],
                doc_padding=retriever_configs.get("doc_padding", "max_length"),
                alignment=retriever_configs["alignment"],
                max_length=retriever_configs.get("max_length", 512),
                subword_weights=retriever_configs.get("subword_weights", 1.),
                filter_stopwords=retriever_configs.get("filter_stopwords", False),
                ot_configs=retriever_configs.get("otp_configs", None)
            )
            retrieval_results = predict_colbert(test_dataset, model)
        elif retriever_configs["model_name"] in ["e5-mistral", "contriever", "bge"]:
            model, tokenizer, _ = load_checkpoint_tokenizer_configs(
                checkpoint_path=retriever_configs["model_path"],
                model_type=retriever_configs["model_type"],
                peft_model=retriever_configs.get("peft_model", False),
                device="cuda"
            )
            model.eval()
            retrieval_results = predict_feature_distance(
                test_dataset, model, tokenizer,
                use_remote_encode=retriever_configs.get("use_remote_encode", False),
                batch_size=retriever_configs.get("batch_size", 8),
                max_length=retriever_configs["max_length"],
                prompt=retriever_configs.get("prompt", None),
                embeddings_method=retriever_configs.get("embeddings_method", None),
                add_eos_token=retriever_configs.get("add_eos_token", False),
                normalize=retriever_configs.get("normalize", False),
            )
        elif retriever_configs["model_name"] in ["splade"]:
            model = SpladeHF(
                model_path=retriever_configs["model_path"],
                device="cuda"
            )
            retrieval_results = predict_feature_distance(
                test_dataset, model, model.tokenizer,
                use_remote_encode=retriever_configs.get("use_remote_encode", False),
                batch_size=retriever_configs.get("batch_size", 8),
                max_length=retriever_configs["max_length"],
                prompt=retriever_configs.get("prompt", None),
                embeddings_method=retriever_configs.get("embeddings_method", None),
                add_eos_token=retriever_configs.get("add_eos_token", False),
                normalize=retriever_configs.get("normalize", False),
            )
        else:
            raise ValueError(retriever_configs["model_name"])
    return retrieval_results


def retrieval_bm25(
    dataset_path,
    dataset_name,
    eval_segment,
    indexes_dir=None,
    k1=None,
    b=None,
    **kwargs
):
    case_dirs, _ = get_segment_data(dataset_path, dataset_name, eval_segment)

    searcher = LuceneSearcher(indexes_dir)
    if k1 and b:
        searcher.set_bm25(k1, b)

    query_scores = {}
    for case_dir in tqdm(case_dirs, desc="BM25 Retrieval"):
        query_id = os.path.split(case_dir)[1]
        query_text = load_paragraph(os.path.join(case_dir, "entailed_fragment.txt"))

        doc_scores = defaultdict(lambda: 0)
        segments = [query_text]
        for segment in segments:
            hits = searcher.search(segment, k=10000)
            for hit in hits:
                if hit.docid.endswith("task2"): 
                    if hit.docid.split("_doc")[0] == query_id:
                        doc_id = hit.docid.split("_task2")[0].split("_doc")[1]
                        doc_scores[doc_id] = aggregate_op(hit.score, doc_scores[doc_id], "max")
        doc_scores = sort_dict(doc_scores, reverse=True)
        query_scores[query_id] = doc_scores
    return query_scores


def run_reranking(
    dataset_path,
    dataset_name,
    eval_segment,
    retrieval_results,
    retrieval_top_k=None,
    model_path=None,
    model_type=None,
    **reranking_configs
):
    model, tokenizer, configs = load_checkpoint_tokenizer_configs(
        checkpoint_path=model_path,
        model_type=model_type,
        device="cuda",
        **reranking_configs,
    )

    query_retrieval_dict = None
    if retrieval_top_k is not None:
        retrieval_results_input, query_retrieval_dict = {}, {}
        for query_id, doc_scores in retrieval_results.items():
            sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            retrieval_results_input[query_id] = {x[0]: x[1] for x in sorted_scores[:retrieval_top_k]}
            query_retrieval_dict[query_id] = list(retrieval_results_input[query_id].keys())
    test_dataset = LegalCaseEntailmentDataset(
        dataset_path, dataset_name, segment=eval_segment,
        model_type=configs["model_type"], query_retrieval_dict=query_retrieval_dict
    )

    if configs["model_type"] == "FEATURE_EXTRACTION":
        scores = predict_feature_distance(
            test_dataset, model, tokenizer, **reranking_configs
        )
    elif configs["model_type"] == "SEQ_CLS":
        scores = predict_seq_cls(test_dataset, model, tokenizer)
    elif configs["model_type"] == "T5":
        scores = predict_monoT5(test_dataset, model)
    else:
        raise ValueError(configs["model_type"])
    return scores


def evaluate(
    dataset_path,
    dataset_name,
    eval_segment,
    retrieval_config_path,
    reranking_config_path=None,
    recompute_retrieval=True,
    recompute_reranking=True,
    top_k=5,
    margin=100,
    threshold=-1
):
    print(dataset_name, eval_segment)
    _, labels = get_segment_data(dataset_path, dataset_name, eval_segment)

    retriever_configs = load_json(retrieval_config_path)
    retriever_signature = get_filename(retrieval_config_path)
    retriever_predict_file = (
        f"./artifacts/{dataset_name}/{eval_segment}/outputs/retrieval/" +
        f"{retriever_configs['model_name']}/{retriever_signature}.json"
    )
    if os.path.exists(retriever_predict_file) and not recompute_retrieval:
        print(f"Load retrieval from ", retriever_predict_file)
        retrieval_results = load_json(retriever_predict_file)
    else:
        print(retrieval_config_path)
        retrieval_results = run_retrieval(
            dataset_path, dataset_name, eval_segment, **retriever_configs
        )
        save_json(retriever_predict_file, retrieval_results, makes_dir=True)

    if reranking_config_path:
        reranking_configs = load_json(reranking_config_path)
        reranker_signature = get_filename(reranking_config_path)
        reranker_predict_file = (
            f"./artifacts/{dataset_name}/{eval_segment}/outputs/prediction/" +
            f"{reranking_configs['model_name']}/{retriever_signature}_{reranker_signature}.json"
        )
        if os.path.exists(reranker_predict_file) and not recompute_reranking:
            print("Load prediction from ", reranker_predict_file)
            reranking_results = load_json(reranker_predict_file)
        else:
            print(reranking_config_path)
            reranking_results = run_reranking(
                dataset_path, dataset_name, eval_segment, retrieval_results, **reranking_configs
            )
            save_json(reranker_predict_file, reranking_results, makes_dir=True)
    else:
        reranker_signature = ""
        reranking_results = None

    if not isinstance(margin, list):
        margin = [margin]
    if not isinstance(top_k, list):
        top_k = [top_k]
    if not isinstance(threshold, list):
        threshold = [threshold]

    config_grids = list(itertools.product(*[top_k, margin, threshold]))
    results = {
        "retrieval_config": retrieval_config_path,
        "reranking_config": reranking_config_path,
        "best": None,
    }

    metrics = [F1Score()]
    best_metrics, best_config = {}, {}
    prediction, best_prediction = {}, {}
    for k, m, thresh in config_grids:
        n_pred = 0
        for query_id, gold_list in labels.items():
            if reranking_results is not None:
                query_score = reranking_results[query_id]
            else:
                query_score = retrieval_results[query_id]
            query_score = {k + ".txt": v for k, v in query_score.items()}

            sorted_score = sorted(query_score.items(), key=lambda x: x[1], reverse=True)[:k]
            pred = [sorted_score[0][0]] if len(sorted_score) else []

            for doc, score in sorted_score[1:]:
                if query_score[pred[0]] - score < m and score >= thresh:
                    pred.append(doc)

            n_pred += len(pred)
            for i in range(len(metrics)):
                metrics[i].add(reference=gold_list, prediction=pred)
            prediction[query_id] = pred

        conf_results = {"avg_n_pred": n_pred / len(prediction)}
        for i in range(len(metrics)): 
            conf_results = {**conf_results, **metrics[i].compute()}
        results[f"k={k}_m={m}_thresh={thresh}"] = conf_results

        if conf_results["f1"] > best_metrics.get("f1", -1):
            best_metrics = conf_results
            best_config = [k, m, thresh]
            best_prediction = prediction

    results["best"] = {
        "metrics": best_metrics,
        "config": best_config
    }
    print(f"Results: ", results["best"])

    best_prediction = sorted(best_prediction.items(), key=lambda x: int(x[0]))
    best_prediction = {k: v for k, v in best_prediction}

    cache_key = (
        f"{retriever_configs['model_name']}" +
        (f"_{reranking_configs['model_name']}" if reranker_signature else "") +
        f"/{retriever_signature}" +
        (f"_{reranker_signature}" if reranker_signature else "")
    )
    pred_save_path = f"./artifacts/{dataset_name}/{eval_segment}/outputs/pipeline/{cache_key}.json"
    res_save_path = f"./artifacts/{dataset_name}/{eval_segment}/results/pipeline/{cache_key}.json"
    save_json(pred_save_path, best_prediction, makes_dir=True)
    save_json(res_save_path, results, makes_dir=True)
    return results["best"]


if __name__ == "__main__":
    fire.Fire(evaluate)
