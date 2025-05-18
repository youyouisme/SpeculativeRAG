import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

### Model name to type mapping
def map_model_type(model_name):
    model_mapping = {
        "gpt-4o-mini": "OpenAI",
        "gpt-3.5-turbo": "OpenAI",
        "Qwen/Qwen2.5-7B": "Qwen",
        "Qwen/Qwen2.5-14B": "Qwen",
        "Qwen/Qwen2.5-32B": "Qwen",
        "Qwen/Qwen2.5-72B": "Qwen",
        "meta-llama/Meta-Llama-3-8B-Instruct": "LLaMA",
        "meta-llama/Meta-Llama-3-70B-Instruct": "LLaMA",
    }
    model_type = model_mapping.get(model_name)
    if not model_type:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model_type

### Model loading
def initialize_model(model_type, model_name):
    if model_type == "OpenAI":
        return model_name, None
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return model, tokenizer

### Unified generation interface
def generate_answer(messages, model, tokenizer, model_type, output_tokens):
    if model_type == "OpenAI":
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=output_tokens,
            temperature=0
        )
        return response.choices[0].message.content.strip()

    elif model_type == "Qwen":
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=output_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    elif model_type == "LLaMA":
        prompt = messages[-1]["content"] if messages else ""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=output_tokens, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()

    else:
        raise ValueError(f"Unsupported model type: {model_type}")