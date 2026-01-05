import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_TEMPLATE = """Summarize the following medical text in 1-2 sentences.

Text:
{input}

Summary:
"""

with open("eval_data.json") as f:
    eval_data = json.load(f)


def load_model(model_name, quantization):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quantization == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    elif quantization == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto"
        )

    elif quantization == "int4":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto"
        )

    else:
        raise ValueError("Quantization must be one of: fp16, int8, int4")

    model.eval()
    return tokenizer, model


def measure_latency(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    torch.cuda.synchronize()
    end = time.time()

    return end - start
