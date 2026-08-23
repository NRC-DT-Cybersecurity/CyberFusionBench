import argparse
import json
import gc
import os
import re
import time
import torch
import pandas as pd
from tqdm import tqdm

from model_utils import (
    load_model_and_tokenizer,
    generate_answer,
)

MODEL_BASE_DIR = os.environ.get("MODEL_BASE_DIR", "")

def model_path(name: str) -> str:
    return os.path.join(MODEL_BASE_DIR, name) if MODEL_BASE_DIR else name

MODELS = {
    "Llama-Primus-Merged-Log-LoRA":
        model_path("Llama-Primus-Merged-Log-LoRA"),
}

BENCHMARK_PATH = os.environ.get("BENCHMARK_PATH", "log_2.json")
CSV_OUTPUT = "log_benchmark_results.csv"

CLASSES = ["Valid", "OsCommanding", "PathTransversal", "XPathInjection", "LdapInjection", "SqlInjection", "SSI", "XSS"]

def extract_class_from_response(response: str) -> tuple[str, str | None]:

    try:

        json_match = re.search(r'\{[^{}]*"choice"[^{}]*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            choice = data.get("choice", "").strip()
            rationale = data.get("rationale", None)

            for cls in CLASSES:
                if cls.lower() == choice.lower():
                    return cls, rationale

            for cls in CLASSES:
                if cls.lower() in choice.lower():
                    return cls, rationale

            return choice if choice else "Unknown", rationale
    except (json.JSONDecodeError, AttributeError):
        pass

    response_lower = response.lower()

    for cls in CLASSES:
        if cls.lower() in response_lower:
            return cls, None

    if "valid" in response_lower:
        return "Valid", None
    elif "attack" in response_lower:
        return "Attack (Ambiguous)", None

    return "Unknown", None

def load_samples(benchmark_path: str) -> list:

    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "samples" in data:
            return data["samples"]
        else:
            raise ValueError("Unknown JSON structure")

def evaluate_log_classification(
    model,
    tokenizer,
    device: str,
    model_name: str,
    samples: list,
    max_samples: int | None = None,
    max_new_tokens: int = 256,
) -> dict:
    if max_samples is not None:
        samples = samples[:max_samples]

    results = []

    ident_correct = 0
    TP = FP = TN = FN = 0

    class_correct = 0
    total = 0

    safe_name = model_name.replace("/", "_")
    live_results_file = f"log_live_{safe_name}.json"

    start_time = time.time()
    pbar = tqdm(samples, desc="Log Classification")

    for idx, sample in enumerate(pbar):
        log_content = sample.get("log", "")
        benchmark_prompt = sample.get("prompt", "")
        ground_truth_class = sample.get("class", "Valid")

        gold_binary = "valid" if ground_truth_class == "Valid" else "attack"

        try:

            response = generate_answer(
                model,
                tokenizer,
                benchmark_prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )

            pred_class, rationale = extract_class_from_response(response)
            pred_binary = "valid" if pred_class == "Valid" else "attack"

            if pred_binary == gold_binary:
                ident_correct += 1

            if gold_binary == "attack" and pred_binary == "attack":
                TP += 1
            elif gold_binary == "valid" and pred_binary == "attack":
                FP += 1
            elif gold_binary == "valid" and pred_binary == "valid":
                TN += 1
            elif gold_binary == "attack" and pred_binary == "valid":
                FN += 1

            is_class_correct = (pred_class == ground_truth_class)
            if is_class_correct:
                class_correct += 1

            total += 1

            curr_ident_acc = ident_correct / total
            curr_class_acc = class_correct / total
            pbar.set_postfix({
                "IdAcc": f"{curr_ident_acc:.2%}",
                "ClAcc": f"{curr_class_acc:.2%}",
                "TP": TP, "FP": FP
            })

            result_entry = {
                "id": idx,
                "log": log_content,
                "raw_output": response,
                "prediction_class": pred_class,
                "prediction_binary": pred_binary,
                "rationale": rationale,
                "gold_class": ground_truth_class,
                "gold_binary": gold_binary,
                "correct_class": is_class_correct,
                "correct_binary": (pred_binary == gold_binary),
            }
            results.append(result_entry)

            precision = TP / (TP + FP) if (TP + FP) else 0.0
            recall = TP / (TP + FN) if (TP + FN) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

            live_data = {
                "model": model_name,
                "progress": f"{total}/{len(samples)}",
                "current_ident_accuracy": round(curr_ident_acc, 4),
                "current_class_accuracy": round(curr_class_acc, 4),
                "current_precision": round(precision, 4),
                "current_recall": round(recall, 4),
                "current_f1": round(f1, 4),
                "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                "ident_correct": ident_correct,
                "class_correct": class_correct,
                "results": results,
            }
            with open(live_results_file, "w", encoding="utf-8") as f:
                json.dump(live_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            total += 1
            results.append({
                "id": idx,
                "log": log_content,
                "raw_output": f"ERROR: {e}",
                "prediction_class": "Error",
                "prediction_binary": "unknown",
                "gold_class": ground_truth_class,
                "gold_binary": gold_binary,
                "correct_class": False,
                "correct_binary": False,
            })

    elapsed_time = time.time() - start_time

    ident_accuracy = ident_correct / total if total else 0.0
    class_accuracy = class_correct / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "total": total,
        "ident_correct": ident_correct,
        "ident_accuracy": round(ident_accuracy, 4),
        "class_correct": class_correct,
        "class_accuracy": round(class_accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "elapsed_time": round(elapsed_time, 2),
        "results": results,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=BENCHMARK_PATH,
        help="Path to benchmark JSON file",
    )
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

    if not os.path.exists(args.benchmark_path):
        print(f"Benchmark file not found: {args.benchmark_path}")
        return

    samples = load_samples(args.benchmark_path)
    print(f"Loaded {len(samples)} samples from benchmark.")

    benchmark_summary = []

    print(f"Starting Log Classification evaluation of {len(MODELS)} models...")

    for model_name, model_path in tqdm(MODELS.items(), desc="Models"):

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

            metrics = evaluate_log_classification(
                model,
                tokenizer,
                device=device,
                model_name=model_name,
                samples=samples,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
            )

            summary_entry = {
                "Model Name": model_name,
                "Model Path": model_path,
                "Ident Accuracy": metrics["ident_accuracy"],
                "Class Accuracy": metrics["class_accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "TN": metrics["TN"],
                "FN": metrics["FN"],
                "Total": metrics["total"],
                "Time (s)": metrics["elapsed_time"],
            }
            benchmark_summary.append(summary_entry)

            safe_name = model_name.replace("/", "_")
            detailed_filename = f"log_results_{safe_name}.json"
            with open(detailed_filename, "w", encoding="utf-8") as f:
                json.dump({
                    "model": model_name,
                    "metrics": {k: v for k, v in metrics.items() if k != "results"},
                    "results": metrics["results"]
                }, f, indent=2, ensure_ascii=False)
            print(f"Saved detailed results to {detailed_filename}")

            print(f"\n----- Results for {model_name} -----")
            print(f"Time: {metrics['elapsed_time']:.2f}s")
            print(f"Ident Accuracy: {metrics['ident_accuracy']:.4f}")
            print(f"Class Accuracy: {metrics['class_accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
            print(f"TP: {metrics['TP']} | FP: {metrics['FP']} | TN: {metrics['TN']} | FN: {metrics['FN']}")

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
                "Ident Accuracy": "ERROR",
                "Class Accuracy": "ERROR",
                "Precision": "ERROR",
                "Recall": "ERROR",
                "F1": "ERROR",
                "TP": 0, "FP": 0, "TN": 0, "FN": 0,
                "Total": 0,
                "Time (s)": 0,
            })

    df_results = pd.DataFrame(benchmark_summary)
    df_results.to_csv(CSV_OUTPUT, index=False)
    print(f"\nSaved benchmark summary to {CSV_OUTPUT}")
    print(df_results)

if __name__ == "__main__":
    main()
