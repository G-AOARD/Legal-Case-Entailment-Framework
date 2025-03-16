import fire

from src.coliee_dataset import LegalCaseEntailmentDataset
from src.retrieval.colbert_ret import ColBERT, predict_colbert
from src.prediction.monoT5 import predict_monoT5
from src.utils.model_utils import load_checkpoint_tokenizer_configs
from src.utils import load_json, save_json, get_filename


def main(
    dataset_path,
    model_configs_file,
    sampling_model="colbert"
):
    config_name = get_filename(model_configs_file)
    model_configs = load_json(model_configs_file)

    dataset = LegalCaseEntailmentDataset(
        dataset_path, "coliee_task2_2024", segment="train",
        model_type=model_configs["model_type"]
    )

    if sampling_model == "colbert":
        model = ColBERT(
            model_configs["model_path"],
            doc_padding=model_configs.get("doc_padding", "max_length"),
            alignment=model_configs["alignment"],
            max_length=model_configs.get("max_length", 512),
            subword_weights=model_configs.get("subword_weights", 1.),
            filter_stopwords=model_configs.get("filter_stopwords", False),
            ot_configs=model_configs.get("otp_configs", None)
        )
        scores = predict_colbert(dataset, model)
    elif sampling_model == "monoT5":
        model, _, _ = load_checkpoint_tokenizer_configs(
            checkpoint_path=model_configs["model_path"],
            model_type=model_configs["model_type"],
            peft_model=model_configs.get("peft_model", False),
            device="cuda"
        )
        scores = predict_monoT5(
            dataset, 
            model
        )

    n_negatives = 20
    negative_samples = {}
    for query_id, gold_list in dataset.labels.items():
        query_scores = sorted(scores[query_id].items(), key=lambda x: x[1], reverse=True)
        docs = [x[0] for x in query_scores]
        negative_docs = [x for x in docs if x + ".txt" not in gold_list][:n_negatives]
        negative_samples[query_id] = negative_docs
    
    save_json(f"./artifacts/negative_samples/{config_name}.json", negative_samples, makes_dir=True)


if __name__ == "__main__":
    fire.Fire(main)
