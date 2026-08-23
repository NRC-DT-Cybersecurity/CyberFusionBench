import argparse
import json
import gc
import os
import re
import torch
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Any, Dict
from tqdm import tqdm

from model_utils import (
    load_model_and_tokenizer,
    generate_answer,
)

MODEL_BASE_DIR = os.environ.get("MODEL_BASE_DIR", "")

def model_path(name: str) -> str:
    return os.path.join(MODEL_BASE_DIR, name) if MODEL_BASE_DIR else name

MODELS = {
    "Llama-Primus-Merged":
        model_path("Llama-Primus-Merged"),

    "microsoft/Phi-4-mini-instruct":
        model_path("Phi-4-mini-instruct"),

     "Qwen/Qwen3-1.7B":
        model_path("Qwen3-1.7B"),

    "internlm/internlm3-8b-instruct":
        model_path("internlm3-8b-instruct"),

    "mistralai/Ministral-3-8B-Instruct-2512":
        model_path("Ministral-3-8B-Instruct-2512"),

    "Qwen/Qwen3-8B":
        model_path("Qwen3-8B"),

    "Foundation-Sec-8B-Instruct":
        model_path("Foundation-Sec-8B-Instruct"),

    "meta-llama/Llama-3.1-8B-Instruct":
        model_path("Llama-3.1-8B-Instruct"),

    "meta-llama/Llama-3.2-1B-Instruct":
        model_path("Llama-3.2-1B-Instruct"),

    "meta-llama/Llama-3.2-3B-Instruct":
        model_path("Llama-3.2-3B-Instruct"),

    "mistralai/Ministral-3-3B-Instruct-2512-BF16":
        model_path("Ministral-3-3B-Instruct-2512-BF16"),

    "Qwen/Qwen3-4B-Instruct-2507":
        model_path("Qwen3-4B-Instruct-2507"),
}

DATA_FILE = "final_benchmark_400.jsonl"
CSV_OUTPUT = "jitvul_benchmark_results_2.csv"

@dataclass
class VulPair:
    f_vul: str
    f_ben: str
    cwe: Optional[str]

def clean_code(code: str) -> str:
    code = code.lstrip()
    if code.startswith("c\n"):
        code = code[2:]
    return code.rstrip()

def load_vul_pairs(path: str, max_pairs: Optional[int] = None) -> List[VulPair]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)

            f_vul = clean_code(obj["vulnerable_function_body"])
            f_ben = clean_code(obj["non_vulnerable_function_body"])

            cwe_list = obj.get("cwe", None)
            if isinstance(cwe_list, list) and len(cwe_list) > 0:
                cwe = cwe_list[0]
            else:
                cwe = None

            pairs.append(VulPair(f_vul=f_vul, f_ben=f_ben, cwe=cwe))

            if max_pairs is not None and i + 1 >= max_pairs:
                break

    return pairs

VANILLA_PROMPT = """You are a security researcher tasked with identifying vulnerabilities in a codebase. You have been given a function to analyze. The function may or may not be vulnerable.

If you think it is vulnerable reply with @@VULNERABLE@@, otherwise reply with @@NOT VULNERABLE@@

If you think the function is vulnerable, please provide the CWE number that you think is most relevant to the vulnerability in the form of @@CWE: <CWE_NUMBER>@@

For example:

@@VULNERABLE@@
@@CWE: CWE-1234@@

Here is the function:

```
{target}
```"""

def build_prompt(code: str) -> str:
    return VANILLA_PROMPT.format(target=code)

def parse_vul_decision(text: str) -> Optional[bool]:
    idx_vul = text.find("@@VULNERABLE@@")
    idx_not = text.find("@@NOT VULNERABLE@@")

    if idx_vul == -1 and idx_not == -1:
        low = text.lower()
        if "not vulnerable" in low:
            return False
        if "vulnerable" in low:
            return True
        return None

    if idx_vul == -1:
        return False
    if idx_not == -1:
        return True

    return idx_vul < idx_not

def parse_cwe_numbers(text: str) -> List[str]:

    numbers = set()

    cwe_matches = re.findall(r"CWE[-:\s]*(\d+)", text, flags=re.I)
    numbers.update(cwe_matches)

    all_numbers = re.findall(r"\b(\d{2,4})\b", text)
    for num in all_numbers:
        if 1 <= int(num) <= 1999:
            numbers.add(num)

    return list(numbers)

def check_cwe_match(predicted_numbers: List[str], ground_truth_cwe: Optional[str]) -> bool:

    if not ground_truth_cwe or not predicted_numbers:
        return False

    gt_match = re.search(r"(\d+)", ground_truth_cwe)
    if not gt_match:
        return False
    gt_number = gt_match.group(1)
    return gt_number in predicted_numbers

def evaluate_jitvul(
    model,
    tokenizer,
    device: str,
    model_name: str,
    pairs: List[VulPair],
    max_new_tokens: int = 256,
) -> dict:

    TP = FP = FN = 0

    correct_pairs = 0
    total_pairs = len(pairs)

    total_cwe = 0
    correct_cwe = 0

    results_log: List[Dict[str, Any]] = []

    safe_name = model_name.replace("/", "_")
    live_results_file = f"jitvul_live_{safe_name}.json"

    for idx, pair in enumerate(tqdm(pairs, desc="JitVul Evaluation")):
        try:

            prompt_vul = build_prompt(pair.f_vul)
            resp_vul = generate_answer(
                model, tokenizer, prompt_vul,
                device=device, max_new_tokens=max_new_tokens, temperature=0.0
            )
            pred_vul = parse_vul_decision(resp_vul)

            prompt_ben = build_prompt(pair.f_ben)
            resp_ben = generate_answer(
                model, tokenizer, prompt_ben,
                device=device, max_new_tokens=max_new_tokens, temperature=0.0
            )
            pred_ben = parse_vul_decision(resp_ben)

            if pred_vul is True:
                TP += 1
            else:
                FN += 1

            if pred_ben is True:
                FP += 1

            is_pairwise_correct = False
            if pred_vul is True and pred_ben is False:
                correct_pairs += 1
                is_pairwise_correct = True

            pred_cwe_numbers = parse_cwe_numbers(resp_vul)
            cwe_correct = False
            if pair.cwe:
                total_cwe += 1
                if check_cwe_match(pred_cwe_numbers, pair.cwe):
                    correct_cwe += 1
                    cwe_correct = True

            results_log.append({
                "id": idx + 1,
                "ground_truth_cwe": pair.cwe,
                "pairwise_correct": is_pairwise_correct,
                "vulnerable_case": {
                    "prompt": prompt_vul,
                    "response": resp_vul,
                    "prediction_is_vul": pred_vul,
                    "predicted_cwes": pred_cwe_numbers,
                    "cwe_correct": cwe_correct
                },
                "benign_case": {
                    "prompt": prompt_ben,
                    "response": resp_ben,
                    "prediction_is_vul": pred_ben
                }
            })

            F1_live = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
            pAcc_live = correct_pairs / (idx + 1)
            cwe_acc_live = correct_cwe / total_cwe if total_cwe > 0 else 0

            live_data = {
                "model": model_name,
                "progress": f"{idx + 1}/{total_pairs}",
                "current_F1": round(F1_live, 4),
                "current_pAcc": round(pAcc_live, 4),
                "current_cwe_accuracy": round(cwe_acc_live, 4),
                "TP": TP, "FP": FP, "FN": FN,
                "correct_pairs": correct_pairs,
                "cwe_correct": correct_cwe,
                "cwe_total": total_cwe,
                "results": results_log,
            }
            with open(live_results_file, "w", encoding="utf-8") as f:
                json.dump(live_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error processing pair {idx}: {e}")
            results_log.append({
                "id": idx + 1,
                "error": str(e),
                "ground_truth_cwe": pair.cwe,
                "pairwise_correct": False,
            })

    F1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    pAcc = correct_pairs / total_pairs if total_pairs > 0 else 0
    cwe_acc = correct_cwe / total_cwe if total_cwe > 0 else 0

    return {
        "total_pairs": total_pairs,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "F1": round(F1, 4),
        "pAcc": round(pAcc, 4),
        "cwe_correct": correct_cwe,
        "cwe_total": total_cwe,
        "cwe_accuracy": round(cwe_acc, 4),
        "results": results_log,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        type=str,
        default=DATA_FILE,
        help="Path to JitVul benchmark JSONL file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda' or 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--max-pairs",
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

    if not os.path.exists(args.data_file):
        print(f"Data file not found: {args.data_file}")
        return

    pairs = load_vul_pairs(args.data_file, max_pairs=args.max_pairs)
    print(f"Loaded {len(pairs)} pairs from benchmark.")

    benchmark_summary = []

    print(f"Starting JitVul evaluation of {len(MODELS)} models...")

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

            metrics = evaluate_jitvul(
                model,
                tokenizer,
                device=device,
                model_name=model_name,
                pairs=pairs,
                max_new_tokens=args.max_new_tokens,
            )

            summary_entry = {
                "Model Name": model_name,
                "Model Path": model_path,
                "F1": metrics["F1"],
                "pAcc": metrics["pAcc"],
                "CWE Accuracy": metrics["cwe_accuracy"],
                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "Total Pairs": metrics["total_pairs"],
            }
            benchmark_summary.append(summary_entry)

            safe_name = model_name.replace("/", "_")
            detailed_filename = f"jitvul_results_{safe_name}.json"
            with open(detailed_filename, "w", encoding="utf-8") as f:
                json.dump({
                    "model": model_name,
                    "metrics": {k: v for k, v in metrics.items() if k != "results"},
                    "results": metrics["results"]
                }, f, indent=2, ensure_ascii=False)
            print(f"Saved detailed results to {detailed_filename}")

            print(f"\n----- Results for {model_name} -----")
            print(f"F1 Score: {metrics['F1']:.4f}")
            print(f"Pairwise Accuracy: {metrics['pAcc']:.4f}")
            print(f"CWE Accuracy: {metrics['cwe_accuracy']:.4f}")
            print(f"TP: {metrics['TP']} | FP: {metrics['FP']} | FN: {metrics['FN']}")

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
                "F1": "ERROR",
                "pAcc": "ERROR",
                "CWE Accuracy": "ERROR",
                "TP": 0, "FP": 0, "FN": 0,
                "Total Pairs": 0,
            })

    df_results = pd.DataFrame(benchmark_summary)
    df_results.to_csv(CSV_OUTPUT, index=False)
    print(f"\nSaved benchmark summary to {CSV_OUTPUT}")
    print(df_results)

if __name__ == "__main__":
    main()
