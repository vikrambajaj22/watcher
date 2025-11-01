"""Loads the OpenAI Open Source Software (OSS) model for inference."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def load_oss_model(model_name="openai/gpt-oss-20b"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = get_device()
    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer
