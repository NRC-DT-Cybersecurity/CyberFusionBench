import os
import math
import logging
import json

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

DATASET_FILE = "jutvil_train_dataset.json"

MODEL_PATH = os.environ.get("MODEL_PATH", "Llama-Primus-Merged")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "lora_jutvil_output")
CACHE_DIR = os.environ.get("CACHE_DIR", "lora_jutvil_cache")

MAX_SEQ_LEN = 2048
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
NUM_EPOCHS = 2
LR_SCHEDULER = "cosine"
SEED = 42
LOGGING_STEPS = 25
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

NUM_PROC = os.cpu_count() or 4
IGNORE_INDEX = -100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_lora_jutvil")

def load_dataset_from_json(path: str) -> Dataset:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Run `python prepare_jutvil_dataset.py` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return Dataset.from_dict({
        "prompt": [r["prompt"] for r in records],
        "response": [r["response"] for r in records],
    })

def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    logger.info("Loading tokenizer from %s", MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        model_max_length=MAX_SEQ_LEN,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model in bf16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Attaching LoRA adapters")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info("Loading dataset from %s ...", DATASET_FILE)
    raw_dataset = load_dataset_from_json(DATASET_FILE)
    logger.info("Dataset has %d examples. Tokenizing ...", len(raw_dataset))

    def tokenize_and_mask(examples):
        all_input_ids = []
        all_labels = []
        all_attention_masks = []

        for prompt, response in zip(examples["prompt"], examples["response"]):

            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            )

            full_ids = tokenizer(
                full_text,
                add_special_tokens=False,
            ).input_ids

            if len(full_ids) > MAX_SEQ_LEN:
                continue

            prompt_ids = tokenizer(
                prompt_only,
                add_special_tokens=False,
            ).input_ids
            prompt_len = min(len(prompt_ids), len(full_ids))

            labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]

            labels = labels[:len(full_ids)]

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            all_attention_masks.append([1] * len(full_ids))

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_masks,
        }

    tokenized_dataset = raw_dataset.map(
        tokenize_and_mask,
        batched=True,
        num_proc=1,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing with chat template",
    )

    total_packed = len(tokenized_dataset)
    logger.info(
        "Tokenized dataset: %d / %d examples kept (fits within %d tokens)",
        total_packed, len(raw_dataset), MAX_SEQ_LEN,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=8,
    )

    total_steps = math.ceil(
        total_packed / (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION)
    ) * NUM_EPOCHS

    logger.info("Total training steps: %d", total_steps)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=int(total_steps * WARMUP_RATIO),
        lr_scheduler_type=LR_SCHEDULER,
        num_train_epochs=NUM_EPOCHS,
        max_grad_norm=1.0,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        save_strategy="steps",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        torch_compile=False,
        seed=SEED,
        report_to="none",
        run_name="jutvil-lora",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("Starting LoRA Training...")
    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    adapter_output = os.path.join(OUTPUT_DIR, "final_adapter")
    logger.info("Saving adapter to %s", adapter_output)
    model.save_pretrained(adapter_output)
    tokenizer.save_pretrained(adapter_output)

    logger.info("Done!")

if __name__ == "__main__":
    main()
