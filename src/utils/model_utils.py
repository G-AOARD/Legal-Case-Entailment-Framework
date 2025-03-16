import os
import torch

from peft import (
    PeftModel,
    PeftConfig,
    LoraConfig,
    get_peft_model,
    AutoPeftModelForCausalLM,
    prepare_model_for_kbit_training
)
from peft.tuners.lora import LoraLayer

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    BitsAndBytesConfig,
    pipeline
)
try:
    import bitsandbytes as bnb
except Exception as e:
    print(e)

from src.llms.utils import LLM_DICT
from src.utils.tokenizer_utils import load_tokenizer
from src.utils import load_json


def get_model_class(model_type):
    if model_type == "CAUSAL_LM":
        return AutoModelForCausalLM
    elif model_type == "SEQ_CLS":
        return AutoModelForSequenceClassification
    elif model_type == "SEQ_2_SEQ_LM":
        return AutoModelForSeq2SeqLM
    elif model_type in ["T5"]:
        return T5ForConditionalGeneration
    elif model_type in ["auto", "FEATURE_EXTRACTION"]:
        return AutoModel
    else:
        raise ValueError(model_type)


def load_model(configs):
    quantization_config = None
    if configs.get("quantization", None):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=configs.get("load_4bit", False),
            load_in_8bit=configs.get("load_8bit", False),
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=configs["quant_type"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    torch_dtype = (torch.float16 if configs.get("use_fp16", False) else
                   torch.bfloat16 if configs.get("use_bp16", False) else torch.float32)

    model_configs = {
        "load_in_4bit": configs.get("load_4bit", False),
        "load_in_8bit": configs.get("load_8bit", False),
        "quantization_config": quantization_config,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "token": os.getenv("HF_ACCESS_TOKEN")
    }
    if configs["model_type"] == "CAUSAL_LM":
        model = AutoModelForCausalLM.from_pretrained(
            configs["model_name"],
            **model_configs
        )
    elif configs["model_type"] == "SEQ_CLS":
        model = AutoModelForSequenceClassification.from_pretrained(
            configs["model_name"],
            num_labels=configs["num_labels"],
            **model_configs
        )
    elif configs["model_type"] in ["SEQ_2_SEQ_LM", "T5"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            configs["model_name"],
            **model_configs
        )
        if configs["model_type"] == "T5":
            make_tensors_contiguous(model)
    else:
        raise ValueError(configs["model_type"])

    if quantization_config is not None:
        prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=configs.get("gradient_checkpointing", False))
    if configs.get("use_peft", False):
        model = create_peft_model(model, configs)

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    return model


def get_lora_task(mode):
    if mode in ["CAUSAL_LM", "SEQ_CLS", "SEQ_2_SEQ_LM"]:
        return mode
    else:
        raise ValueError(mode)


def create_peft_model(model, configs):
    if configs["peft_method"] == "lora":
        peft_config = LoraConfig(
            r=configs["lora_r"],
            lora_alpha=configs["lora_alpha"],
            target_modules=configs["lora_modules"],
            lora_dropout=configs["lora_dropout"],
            bias="none",
            task_type=get_lora_task(configs["model_type"]),
            inference_mode=False,
        )
    else:
        raise ValueError(configs["peft_method"])

    model = get_peft_model(
        model,
        peft_config=peft_config
    )
    torch_dtype = (torch.float16 if configs.get("use_fp16", False) else
                   torch.bfloat16 if configs.get("use_bp16", False) else torch.float32)
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch_dtype)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module = module.to(torch_dtype)
    return model


def load_checkpoint_tokenizer_configs(
    checkpoint_path,
    model_configs={},
    model_type=None,
    peft_model=False,
    num_labels=None,
    device=None,
    **kwargs
):
    model_dict = LLM_DICT.get(checkpoint_path, {})
    configs = None
    if os.path.exists(checkpoint_path):
        config_path = os.path.join(
            checkpoint_path.split("checkpoints")[0],
            "configs.json"
        )
        if os.path.exists(config_path):
            configs = load_json(config_path)
        else:
            configs = None

    if configs is None:
        if not model_type:
            model_type = model_dict.get("model_type", None)
        assert model_type is not None
        if model_type == "together.ai":
            return None, None, None
        configs = {
            "model_name": checkpoint_path,
            "model_type": model_type,
            "use_peft": peft_model,
        }

    if "torch_dtype" not in model_configs:
        torch_dtype = (torch.float16 if model_dict.get("fp16", False) else
            torch.bfloat16 if model_dict.get("bp16", False) else torch.float32)
        model_configs["torch_dtype"] = torch_dtype
    
    if configs["model_type"] == "PIPELINE":
        model = pipeline(
            "text-generation",
            model=checkpoint_path,
            model_kwargs={"torch_dtype": torch_dtype},
            device=device,
            token=os.getenv("HF_ACCESS_TOKEN"),
        )
        return model, model.tokenizer, configs
    else:
        if configs["model_type"] == "SEQ_CLS":
            model_configs["num_labels"] = num_labels

        if configs.get("use_peft", False):
            model = load_peft_model(
                checkpoint_path, model_type=configs["model_type"], **model_configs)
        else:
            model = get_model_class(configs["model_type"]).from_pretrained(
                checkpoint_path,
                token=os.getenv("HF_ACCESS_TOKEN"),
                trust_remote_code=True,
                **model_configs
            )

        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = model.config.eos_token_id
        if device:
            model = model.to(device)
        model.eval()
        
        try:
            tokenizer = load_tokenizer(configs["model_name"], peft_model=configs.get("use_peft", False))
        except Exception as e:
            tokenizer = None

    return model, tokenizer, configs


def load_peft_model(peft_model_name, model_type=None, **kwargs):
    config = PeftConfig.from_pretrained(peft_model_name)

    base_model = get_model_class(model_type).from_pretrained(
        config.base_model_name_or_path,
        token=os.getenv("HF_ACCESS_TOKEN"),
        **kwargs
    )
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    return model


def merge_weights(model_path, model_name, output_dir):
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if num_new_tokens > 0:
        input_embeddings_data = input_embeddings.weight.data
        output_embeddings_data = output_embeddings.weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg
        model.tie_weights()

    input_embeddings.weight.requires_grad = False
    output_embeddings.weight.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def print_trainable_parameters(model, use_4bit=False):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if use_4bit == 4:
        trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def verify_parameters(model):
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(f"{k}: {v} {v / total}")


def make_tensors_contiguous(model):
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()


def get_max_length(model):
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 4096
        print(f"Using default max length: {max_length}")
    return max_length


def str_or_bool(value):
    if str(value).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(value).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return str(value)


def average_pooling(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def extract_embeddings(
    model,
    tokenizer,
    texts,
    use_remote_encode=False,
    batch_size=8,
    max_length=512,
    input_template="{text}",
    add_eos_token=False,
    embeddings_method=None,
    normalize=False,
):
    if use_remote_encode:
        embeddings = model.encode(texts, max_length=max_length)
    else:
        assert embeddings_method in [
            "pooler_output", "first_token", "last_token", "average_pooling", "last_token_pooling",
            "none"
        ]
        embeddings = []
        if not batch_size:
            batch_size = len(texts)
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            if add_eos_token:
                encoded = tokenizer(
                    [input_template.format_map({"text": text}) for text in batch_texts],
                    max_length=max_length - 1,
                    return_attention_mask=False,
                    padding=False,
                    truncation=True
                )
                encoded['input_ids'] = [input_ids + [tokenizer.eos_token_id]
                                        for input_ids in encoded['input_ids']]
                encoded = tokenizer.pad(
                    encoded, padding=True, return_attention_mask=True, return_tensors='pt')
            else:
                encoded = tokenizer(
                    [input_template.format_map({"text": text}) for text in batch_texts],
                    padding=True,
                    max_length=max_length,
                    truncation="longest_first",
                    return_tensors="pt"
                )

            with torch.no_grad():
                for key in encoded.keys():
                    encoded[key] = encoded[key].to("cuda")
                outputs = model(**encoded)

            if embeddings_method == "pooler_output":
                batch_embeddings = outputs.pooler_output
            elif embeddings_method == "first_token":
                batch_embeddings = outputs.last_hidden_state[:, 0]
            elif embeddings_method == "last_token":
                batch_embeddings = outputs.last_hidden_state[:, -1]
            elif embeddings_method == "average_pooling":
                batch_embeddings = average_pooling(
                    outputs.last_hidden_state, encoded["attention_mask"])
            elif embeddings_method == "last_token_pooling":
                batch_embeddings = last_token_pool(
                    outputs.last_hidden_state, encoded["attention_mask"])
            elif embeddings_method == "none":
                batch_embeddings = outputs
            else:
                raise ValueError(embeddings_method)
            if normalize:
                batch_embeddings = torch.nn.functional.normalize(
                    batch_embeddings, p=2, dim=-1)
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)
