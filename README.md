# Official implementation of the paper: A Retrieval-focused Language Models Framework for Legal Case Entailment
### The system is implemented by Tran Cong Thanh with Prof Nguyen Le Minh (PI, supervisor) in NguyenLab@JAIST

## Results on COLIEE Legal Case Entailment Task

|                    | COLIEE 2020 | COLIEE 2021 | COLIEE 2022 | COLIEE 2023 | COLIEE 2024 |
|--------------------|-------------|-------------|-------------|-------------|-------------|
| Best system        | 67.53       | 69.51       | 67.83       | 74.56       | 65.12       |
| ColBERT - MonoT5   | 68.97       | 75.77       | 74.67       | 77.13       | 69.82       |
   
## Reproduce the results

- ColBERT model weights: `colbert-ir/colbertv2.0`
- Fine-tuned MonoT5 model weights: `thanhtc/monot5-3B-legal-ent`

### [1] ColBERT-UOT retrieval result
```bash
        python src/evaluate.py \
            --dataset_path=<path> \
            --dataset_name=coliee_task2_2024 \
            --eval_segment=test \
            --retrieval_config_path=configs/framework/retrieval/colbert-uot.json \
            --top_k=20
```

### [2] ColBERT-UOT & MonoT5 entailment prediction result
```bash
        python src/evaluate.py \
            --dataset_path=<path> \
            --dataset_name=coliee_task2_2024 \
            --eval_segment=test \
            --retrieval_config_path=configs/framework/retrieval/colbert-uot.json \
            --reranking_config_path=configs/framework/prediction/monoT5-3B.json \
            --top_k=2 \
            --margin=1 \
            --threshold=90
```

### [3] Fine-tuning MonoT5 on COLIEE dataset
```bash
    python src/training/finetune.py \
        configs/train/monoT5-3B.json \
        --dataset_path=<path> \
        --dataset_name=coliee_task2_2024
```

### [4] Zero-shot list-wise entailment prediction with LLM
- Must run [2] first to obtain the re-ranking output of MonoT5
- Look up the prompt list in `src/llms/prompts/legal_entailment.py`
- Lookup the supported LLMs in `src/llms/utils.py`

```bash
    python src/llms/prompt_llm.py \
        --dataset_path=<path> \
        --dataset_name=coliee_task2_2024 \
        --eval_segment=test \
        --reranking_file=artifacts/coliee_task2_2024/test/outputs/prediction/monoT5/colbert-uot_monoT5-3B.json \
        --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
        --prompt_id=lte.pr.sa.1 \
        --max_new_tokens=50 \
        --top_k=5 \
        --use_cpu=False
```

