import os
import re
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_LOCAL_PATH = os.environ.get("DEFAULT_MODEL_PATH", "Foundation-Sec-8B-Instruct")

def validate_model_folder(path: str):

    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"\n❌ ERROR: Model directory does not exist:\n{path}\n"
            f"Please download the model first.\n"
        )

    required_configs = ["config.json"]
    missing = [f for f in required_configs if not os.path.exists(os.path.join(path, f))]

    if missing:
        raise FileNotFoundError(
            f"\n❌ ERROR: Model directory is missing required config files:\n"
            f"{missing}\n"
            f"Directory: {path}\n"
        )

    possible_tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "tokenizer_config.json"
    ]

    has_tokenizer = any(os.path.exists(os.path.join(path, f)) for f in possible_tokenizer_files)

    if not has_tokenizer:
        raise FileNotFoundError(
            f"\n❌ ERROR: No tokenizer file found in:\n{path}\n"
            f"Expected at least one of: {possible_tokenizer_files}\n"
        )

def load_model_and_tokenizer(
    model_path: str | None = None,
    device: str | None = None,
    dtype: str = "bfloat16",
):

    if model_path is None:
        model_path = DEFAULT_LOCAL_PATH

    validate_model_folder(model_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    print(f"✅ Loading model from local path:\n{model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    if not tokenizer.chat_template:
        print("⚠️ Warning: Tokenizer has no chat_template defined. Instruct models might perform poorly.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device != "cuda":
        model.to(device)

    print("✅ Model loaded successfully (local only).")
    return model, tokenizer, device

@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> str:

    messages = [{"role": "user", "content": prompt}]

    if tokenizer.chat_template:
        encodings = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=False
        )
        input_ids = encodings.input_ids.to(model.device)
        attention_mask = encodings.attention_mask.to(model.device)
    else:

        encodings = tokenizer(prompt, return_tensors="pt")
        input_ids = encodings.input_ids.to(model.device)
        attention_mask = encodings.attention_mask.to(model.device)

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def extract_mcq_letter(text: str) -> str | None:

    if not text:
        return None

    text = text.strip()

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return None

    last_line = lines[-1]

    if last_line in {"A", "B", "C", "D"}:
        return last_line

    if last_line.startswith(("A)", "B)", "C)", "D)")):
        return last_line[0]

    match = re.fullmatch(r"\*\*([ABCD])\*\*", last_line)
    if match:
        return match.group(1)

    match = re.search(r"([ABCD])\s*$", last_line)
    if match:
        return match.group(1)

    match = re.search(r"Answer:\s*([ABCD])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None

def extract_cwe_id(text: str) -> str | None:

    matches = re.findall(r"CWE-\d+", text, re.IGNORECASE)
    return matches[-1].upper() if matches else None

def extract_cvss_vector(text: str) -> str | None:

    pattern = r"AV:[A-Z]+(?:/[A-Z]+:[A-Z]+)+"
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None

def extract_attack_ids(text: str) -> List[str]:
    ids = re.findall(r"\bT\d{4}(?:\.\d{3})?\b", text.upper())
    return sorted(set(ids))
