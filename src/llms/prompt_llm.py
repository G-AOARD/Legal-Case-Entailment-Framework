import os
import re
import fire

from tqdm import tqdm

import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

from src.coliee_dataset import get_segment_data, load_paragraph
from src.utils.model_utils import load_checkpoint_tokenizer_configs
from src.llms.utils import llm_generate
from src.llms.prompts.legal_entailment import PROMPT_DICT
from src.training.metrics import F1Score
from src.utils import load_json, save_json


def extract_answers(resp):
    text = resp
    llm_answer = list(set(re.findall(r'P\d+', text)))
    llm_answer = [x[2:] + ".txt" for x in llm_answer if len(x) == 5]
    return list(set(llm_answer))


def main(
    dataset_path,
    dataset_name,
    eval_segment="test",
    reranking_file=None,
    model_name=None,
    prompt_id=None,
    max_length=4000,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.,
    top_k=5,
    recompute=False,
    use_cpu=False,
):
    eval_case_dirs, eval_labels = get_segment_data(dataset_path, dataset_name, eval_segment=eval_segment)

    reranking_filename = os.path.splitext(os.path.split(reranking_file)[1])[0]
    signature_str = f"{reranking_filename}_p={prompt_id}_k={top_k}_mt={max_new_tokens}_sample={do_sample}-t={temperature}.json"
    prediction_file = (f"./artifacts/{dataset_name}/{eval_segment}/outputs/llms/" +
                     f"{model_name.replace('/', '_')}/" + signature_str)
    result_file = (f"./artifacts/{dataset_name}/{eval_segment}/results/llms/" +
                   f"{model_name.replace('/', '_')}/" + signature_str)
    if os.path.exists(prediction_file) and not recompute:
        do_infer = False
        answer_dict = load_json(prediction_file)
    else:
        do_infer = True

        if use_cpu:
            device = "cpu"
        else:
            device = "cuda"

        model, tokenizer, _ = load_checkpoint_tokenizer_configs(
            model_name, device=device
        )
        answer_dict = {}

    reranking_results = load_json(reranking_file)
    top_ranking = {}
    for case_dir in tqdm(eval_case_dirs):
        case_id = os.path.split(case_dir)[1]
        preds = reranking_results[case_id]

        sorted_score = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        top_docs = [(k + ".txt", score) for k, score in sorted_score[:top_k]]
        top_ranking[case_id] = top_docs

        if do_infer:
            query_text = load_paragraph(
                os.path.join(case_dir, "entailed_fragment.txt"), max_length=400
            )
            doc_texts = [
                f"ID P{d.zfill(8)}: " + load_paragraph(os.path.join(case_dir, "paragraphs", d), max_length=400)
                for i, (d, s) in enumerate(top_docs)
            ]

            input_text = PROMPT_DICT[prompt_id]["input"].format_map({
                "docs": "\n\n".join(doc_texts),
                "query": query_text
            }) + PROMPT_DICT[prompt_id]["response"]
            resp = llm_generate(
                model_name, model, tokenizer, input_text, system_text=PROMPT_DICT[prompt_id].get("system", None),
                max_length=max_length, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature,
                device=device
            )
            resp = resp.split(PROMPT_DICT[prompt_id]["response"])[-1]
            answer_dict[case_id] = resp

    if do_infer:
        save_json(prediction_file, answer_dict, makes_dir=True)

    case_candidates = {}
    for case_dir in tqdm(eval_case_dirs):
        case_candidates[os.path.split(case_dir)[1]] = os.listdir(os.path.join(case_dir, "paragraphs"))

    tp, fp, fn = 0, 0, 0
    for case_id, resp in answer_dict.items():
        llm_answer = extract_answers(resp)
        llm_answer = [x for x in llm_answer if x in case_candidates[case_id]]

        tp += sum([1 for doc in llm_answer if doc in eval_labels[case_id]])
        fp += sum([1 for doc in llm_answer if doc not in eval_labels[case_id]])
        fn += sum([1 for doc in eval_labels[case_id] if doc not in llm_answer])

    f1 = F1Score.f1_score(tp, fp, fn)
    save_json(result_file, f1, makes_dir=True)

    print(f"Model: {model_name} - {prompt_id}", )
    print(f"Datasets: {dataset_name} - {eval_segment}")
    print("LLM result: ", f1)

    if do_infer:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    fire.Fire(main)
