import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

llm_dir = "./Qwen3-4B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(llm_dir)
model = AutoModelForCausalLM.from_pretrained(
    llm_dir,
    torch_dtype="auto",
    device_map="auto"
)

def create_chat_completion(messages, temperature, thinking, *, max_new_tokens=64):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        top_p = 0.9
    )
    reply = tokenizer.decode(
        output[0, inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )
    return reply.strip()
