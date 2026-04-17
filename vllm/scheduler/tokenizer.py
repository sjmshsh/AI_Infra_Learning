# 定义tokenizer:
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=True)
prompts = [
    "hi, I'm kaiyuan",
    "Do you subscribe InfraTech?",
]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in prompts
]
for prompt in prompts:
    print(tokenizer.encode(prompt))