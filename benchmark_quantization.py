import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT_TEMPLATE = "Summarize the following text in 1-2 sentences.\n\nText:\n{input}\n\nSummary:\n"

def load_model(model_name, quantization):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantization == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    elif quantization == "int8":
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
    elif quantization == "int4":
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
    return tokenizer, model

def measure_latency(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return time.time() - start
