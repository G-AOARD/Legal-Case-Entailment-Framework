import os
import time
import torch


LLM_DICT = {
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "model_type": "CAUSAL_LM",
        "input_chat_template": True,
        "support_system_prompt": False,
        "end_instruction_token": "[/INST]"
    },
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {
        "model_type": "together.ai",
        "input_chat_template": True,
        "support_system_prompt": False,
        "end_instruction_token": "[/INST]"
    },
    "google/gemma-1.1-7b-it": {
        "model_type": "CAUSAL_LM",
        "input_chat_template": True,
        "support_system_prompt": False,
        "end_instruction_token": "<end_of_turn>"
    },
    "Qwen/Qwen1.5-72B-Chat": {
        "model_type": "together.ai",
        "support_system_prompt": True,
        "end_instruction_token": "<|im_end|>\n<|im_start|>assistant"
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "model_type": "PIPELINE",
        "bp16": True,
        "support_system_prompt": True,
        "input_chat_template": True,
        "add_generation_prompt": True,
    },
    "meta-llama/Llama-3-70b-chat-hf": {
        "model_type": "together.ai",
        "support_system_prompt": True
    }
}


def together_api_call(model_name, input_text, system_text, support_system_prompt, max_new_tokens):
    from together import Together

    messages = create_input_chat_template(input_text, system_text, support_system_prompt)
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0,
        top_k=50,
        top_p=1
    )
    time.sleep(0.1)
    return response.choices[0].message.content


def create_input_chat_template(input_content, system_text, support_system_prompt=False, tokenizer=None,
                               add_generation_prompt=False):
    messages = []
    if system_text:
        if support_system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_text
                }
            )
        else:
            input_content = system_text + input_content

    messages.append(
        {
            "role": "user",
            "content": input_content
        }
    )
    if tokenizer is not None:
        messages = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    return messages


def llm_generate(
    base_model_name,
    model,
    tokenizer,
    input_text,
    system_text=None,
    max_length=None,
    max_new_tokens=None,
    do_sample=False,
    temperature=0.,
    top_k=50,
    top_p=1,
    device=None
):
    if LLM_DICT[base_model_name].get("model_type") == "together.ai":
        answer = together_api_call(
            base_model_name, input_text, system_text, LLM_DICT[base_model_name].get("support_system_prompt", False),
            max_new_tokens
        )
    elif LLM_DICT[base_model_name].get("use_chat_api", False):
        answer, _ = model.chat(tokenizer, input_text, history=None)
    else:
        if LLM_DICT[base_model_name].get("input_chat_template", False):
            input_text = create_input_chat_template(
                input_text, system_text, LLM_DICT[base_model_name].get("support_system_prompt", False),
                tokenizer, add_generation_prompt=LLM_DICT[base_model_name].get("add_generation_prompt", False))
        if LLM_DICT[base_model_name]["model_type"] == "PIPELINE":
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            outputs = model(
                input_text,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=do_sample,
            )
            decoded_text = outputs[0]["generated_text"][len(input_text):]
        else:
            if not LLM_DICT[base_model_name].get("input_chat_template", False):
                input_text = (system_text or "") + input_text
            encoded_input = tokenizer.encode(input_text, add_special_tokens=False)[:max_length]
            encoded_input_tensor = torch.tensor([encoded_input], dtype=torch.int64).to(device)
            generated_ids = model.generate(
                encoded_input_tensor, max_new_tokens=max_new_tokens,
                do_sample=do_sample, temperature=temperature, top_k=top_k,
                top_p=top_p,
            )
            decoded_text = tokenizer.batch_decode(generated_ids)[0]
        if LLM_DICT[base_model_name].get("end_instruction_token", None):
            answer = decoded_text.split(LLM_DICT[base_model_name]["end_instruction_token"])[-1]
        else:
            answer = decoded_text
    return answer.strip()
