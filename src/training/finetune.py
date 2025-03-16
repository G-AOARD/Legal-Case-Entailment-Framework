import os
import sys
import copy
import time
import fire
import shutil

from tqdm import tqdm
from contextlib import nullcontext

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cuda

torch.backends.cudnn.deterministic = True

from transformers import set_seed
from transformers.optimization import Adafactor

from src.coliee_dataset import load_dataset
from src.dataloaders import get_dataloader
from src.prediction.monoT5 import evaluate_monoT5
from src.utils.model_utils import (
    load_model,
    load_tokenizer,
    print_trainable_parameters,
    verify_parameters,
    get_max_length,
)
from src.utils import Logger
from src.utils.file_utils import (
    load_json,
    save_json,
)


def finetune(configs, save_dir):
    seed = configs.get("seed", None)
    if seed is None:
        configs["seed"] = np.random.randint(1, 1e5)
    set_seed(configs["seed"])

    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    log_dir = os.path.join(save_dir, "log")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = load_model(configs)
    if not configs.get("quantization", False):
        model.to("cuda")
    print_trainable_parameters(model, configs.get("load_4bit", False))
    verify_parameters(model)

    context_length = configs.get("context_length", None) or get_max_length(model)
    configs["context_length"] = context_length
    tokenizer = load_tokenizer(configs["model_name"], context_length)
    train_dataset, val_dataset, test_dataset = load_dataset(
        configs["dataset_path"],
        dataset_name=configs["dataset_name"],
        model_type=configs["model_type"],
        word_threshold=configs.get("word_threshold", None),
        max_num_sentences=configs.get("max_num_sentences", None),
        num_positives_per_example=configs.get("num_positives_per_example", None),
        num_negatives_per_example=configs.get("num_negatives_per_example", None),
        sampling_strategy=configs.get("sampling_strategy", None),
        query_retrieval_file=configs.get("query_retrieval_file", None),
        reranking_batching=configs.get("reranking_batching", False),
    )
    train_dataloader = get_dataloader(
        train_dataset,
        configs,
        tokenizer,
        is_train=True
    )

    if configs["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=configs["lr"],
            momentum=configs["momentum"],
            weight_decay=configs["weight_decay"],
        )
    elif configs["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=configs["lr"],
            weight_decay=configs["weight_decay"],
            fused=not(configs.get("load_8bit", False) or configs.get("load_4bit", False)),
        )
    elif configs["optimizer"] == "adafactor":
        optimizer = Adafactor(
            model.parameters(),
            lr=configs["lr"],
            clip_threshold=1.0,
            relative_step=configs.get("relative_step", False),
            scale_parameter=configs.get("scale_parameter", False),
            warmup_init=configs.get("warmup_init", False),
            weight_decay=configs["weight_decay"],
        )
    else:
        raise ValueError(configs["optimizer"])

    autocast = nullcontext
    amp_train = configs.get("use_fp16", False) or configs.get("use_bp16", False)
    if amp_train:
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast

    save_json(os.path.join(save_dir, "configs.json"), configs)

    if isinstance(configs["val_metric"], list):
        selection_metric = configs["val_metric"][0]
    else:
        selection_metric = configs["val_metric"]
    val_results, test_results = {}, {}
    best_val_metric = -float("inf")
    global_step = 0
    for epoch in range(configs["num_epochs"]):
        model.train()
        if epoch > 0:
            train_dataset.build_input_dataset()

        epoch_start_time = time.perf_counter()
        total_loss = 0.0
        total_length = len(train_dataloader) // configs["gradient_accumulation_steps"]
        pbar = tqdm(colour="blue", desc=f"Epoch: {epoch + 1}", total=total_length,
                    dynamic_ncols=True)
        for step, batch in enumerate(train_dataloader):
            for key in batch.keys():
                batch[key] = batch[key].to("cuda")
            with autocast():
                loss = model(**batch).loss
            loss = loss / configs["gradient_accumulation_steps"]
            total_loss += loss.detach().float()
            if configs.get("use_fp16", False):
                scaler.scale(loss).backward()
                if (step + 1) % configs["gradient_accumulation_steps"] == 0 or \
                        step == len(train_dataloader) - 1:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    pbar.update(1)
            else:
                loss.backward()
                if (step + 1) % configs["gradient_accumulation_steps"] == 0 or \
                        step == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)
            global_step += 1
            pbar.set_description(
                f"Epoch: {epoch + 1}/{configs['num_epochs']}, step {step + 1}/{len(train_dataloader)}, "
                f"lr: {optimizer.param_groups[0]['lr']}, loss: {loss.detach().float()}")
        pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)

        epoch_dir = os.path.join(checkpoint_dir, f"epoch-{(epoch + 1)}")
        val_metrics = evaluate_monoT5(test_dataset=val_dataset, model=model)
        print(f"Epoch {epoch + 1}/{configs['num_epochs']}: "
              f"train_perplexity={train_perplexity:.4f}, "
              f"train_epoch_loss={train_epoch_loss:.4f}, "
              f"metrics={val_metrics['best']['metrics']}, "
              f"epoch time {epoch_end_time}s"
        )

        if val_metrics["best"]["metrics"][selection_metric] >= best_val_metric:
            best_val_metric = val_metrics["best"]["metrics"][selection_metric]
            val_results = copy.deepcopy(val_metrics)
            model.save_pretrained(os.path.join(checkpoint_dir, "best"))
            
            if test_dataset is not None:
                test_results = evaluate_monoT5(test_dataset=test_dataset, model=model)
                print(f"[TEST]: {test_results['best']['metrics']}")
        model.save_pretrained(epoch_dir)

    save_json(os.path.join(save_dir, "val_results.json"), val_results)
    save_json(os.path.join(save_dir, "test_results.json"), test_results)

    del model
    torch.cuda.empty_cache()


def main(config_path, dataset_name=None, dataset_path=None, save_dir=None, force=None):
    configs = load_json(config_path)
    configs["dataset_name"] = dataset_name
    configs["dataset_path"] = dataset_path

    if not save_dir:
        config_name = os.path.splitext(os.path.split(config_path)[1])[0]
        save_dir = f"./train_logs/{configs['dataset_name']}/{config_name}"
    if os.path.exists(save_dir) and force:
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    sys.stdout = Logger(
        pipe=sys.stdout, log_path=os.path.join(save_dir, "stdout.txt")
    )
    sys.stderr = Logger(
        pipe=sys.stderr, log_path=os.path.join(save_dir, "stderr.txt")
    )
 
    finetune(configs, save_dir)


if __name__ == "__main__":
    fire.Fire(main)
