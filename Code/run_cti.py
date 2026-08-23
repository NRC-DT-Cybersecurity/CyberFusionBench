import argparse
import json
import gc
import os
import torch
import pandas as pd
from tqdm import tqdm

from model_utils import (
    load_model_and_tokenizer,
    generate_answer,
    extract_mcq_letter,
)

MODEL_BASE_DIR = os.environ.get("MODEL_BASE_DIR", "")

def model_path(name: str) -> str:
    return os.path.join(MODEL_BASE_DIR, name) if MODEL_BASE_DIR else name

MODELS = {

    "Qwen/Qwen3-1.7B":
        model_path("Qwen3-1.7B"),

    "Qwen/Qwen3-8B":
        model_path("Qwen3-8B"),

    "internlm/internlm3-8b-instruct":
        model_path("internlm3-8b-instruct"),

    "Foundation-Sec-8B-Instruct":
        model_path("Foundation-Sec-8B-Instruct"),

    "meta-llama/Llama-3.1-8B-Instruct":
        model_path("Llama-3.1-8B-Instruct"),

    "meta-llama/Llama-3.2-1B-Instruct":
        model_path("Llama-3.2-1B-Instruct"),

    "meta-llama/Llama-3.2-3B-Instruct":
        model_path("Llama-3.2-3B-Instruct"),

    "Llama-Primus-Merged":
        model_path("Llama-Primus-Merged"),

    "microsoft/Phi-4-mini-instruct":
        model_path("Phi-4-mini-instruct"),

    "Qwen/Qwen3-4B-Instruct-2507":
        model_path("Qwen3-4B-Instruct-2507"),
}

MCQ_URL = "https://huggingface.co/datasets/AI4Sec/cti-bench/raw/main/cti-mcq.tsv"
CSV_OUTPUT = "cti_mcq_benchmark_results.csv"

def load_mcq_df() -> pd.DataFrame:
    df = pd.read_csv(MCQ_URL, sep="\t")

    return df

def evaluate_mcq(
    model,
    tokenizer,
    device: str,
    model_name: str,
    max_samples: int | None = None,
    max_new_tokens: int = 256,
) -> dict:
    df = load_mcq_df()
    if max_samples is not None:
        df = df.head(max_samples)

    prompts = df["Prompt"].tolist()
    gold = df["GT"].astype(str).str.strip().tolist()

    safe_name = model_name.replace("/", "_")
    live_results_file = f"mcq_live_{safe_name}.json"

    results = []
    correct_so_far = 0

    for idx, (prompt, g) in enumerate(
        tqdm(zip(prompts, gold), total=len(gold), desc="MCQ")
    ):
        raw_output = generate_answer(
            model,
            tokenizer,
            prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )

        pred = extract_mcq_letter(raw_output)
        pred = pred if pred is not None else "_"

        is_correct = (pred == g)
        if is_correct:
            correct_so_far += 1

        result_entry = {
            "id": idx,
            "prompt": prompt,
            "raw_output": raw_output,
            "prediction": pred,
            "gold": g,
            "correct": is_correct,
        }
        results.append(result_entry)

        current_acc = correct_so_far / len(results)
        live_data = {
            "model": model_name,
            "progress": f"{len(results)}/{len(gold)}",
            "current_accuracy": round(current_acc, 4),
            "correct_so_far": correct_so_far,
            "results": results,
        }
        with open(live_results_file, "w") as f:
            json.dump(live_data, f, indent=2)

    acc = correct_so_far / len(results) if results else 0.0
    return {
        "total": len(results),
        "correct": correct_so_far,
        "accuracy": round(acc, 4),
        "results": results
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda' or 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional subset size for quick testing.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    args = parser.parse_args()

    benchmark_summary = []

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU Memory before start: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")

    print(f"Starting evaluation of {len(MODELS)} models...")

    for model_name, model_path in tqdm(MODELS.items(), desc="Models"):
        print(f"\n==========================================")
        print(f"Loading: {model_name}")
        print(f"Path: {model_path}")
        print(f"==========================================")

        try:
            model, tokenizer, device = load_model_and_tokenizer(
                model_path=model_path,
                device=args.device,
                dtype="bfloat16",
            )

            metrics = evaluate_mcq(
                model,
                tokenizer,
                device=device,
                model_name=model_name,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
            )

            summary_entry = {
                "Model Name": model_name,
                "Model Path": model_path,
                "Accuracy": metrics["accuracy"],
                "Correct": metrics["correct"],
                "Total": metrics["total"]
            }
            benchmark_summary.append(summary_entry)

            safe_name = model_name.replace("/", "_")
            detailed_filename = f"results_new_{safe_name}.json"
            with open(detailed_filename, "w") as f:
                json.dump({
                    "model": model_name,
                    "metrics": metrics
                }, f, indent=2)
            print(f"Saved detailed results to {detailed_filename}")

            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            benchmark_summary.append({
                "Model Name": model_name,
                "Model Path": model_path,
                "Accuracy": "ERROR",
                "Correct": 0,
                "Total": 0
            })

    df_results = pd.DataFrame(benchmark_summary)
    df_results.to_csv(CSV_OUTPUT, index=False)
    print(f"\nSaved benchmark summary to {CSV_OUTPUT}")
    print(df_results)

if __name__ == "__main__":
    main()
