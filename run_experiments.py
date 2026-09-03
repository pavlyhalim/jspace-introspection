"""
Main Experiment Runner: Benchmarking J-Space ICL vs. Continuous Injection vs. Observer
Author: Pavly Halim <pavlyhalim@gmail.com>
"""

import argparse
import json
import os
from typing import Dict, Any, List
import torch
from tabulate import tabulate
from tqdm import tqdm

from jspace.models import load_model, get_layers_and_head
from jspace.benchmark import get_reasoning_benchmark
from jspace.metrics import compute_metrics
from jspace.interventions import causal_ablation_check
from jspace.baselines.observer import run_observer_baseline
from jspace.baselines.pure_icl import run_pure_icl_self_report
from jspace.baselines.continuous_injection import run_continuous_injection_baseline


def parse_args():
    parser = argparse.ArgumentParser(description="J-Space Introspection Benchmark")
    parser.add_argument("--model", type=str, default="llama-3.1-8b",
                        choices=["qwen-2.5-0.5b", "llama-3.2-1b", "llama-3.1-8b", "qwen-2.5-7b", "gemma-4-e2b", "gemma-4-26b-a4b"],
                        help="Target model architecture")
    parser.add_argument("--custom_repo_id", type=str, default=None,
                        help="Custom Hugging Face repository ID")
    parser.add_argument("--target_layer", type=int, default=None,
                        help="Override Global Workspace target ignition layer")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device map ('auto', 'cuda', 'mps', 'cpu')")
    parser.add_argument("--ablation_strength", type=float, default=3.0,
                        help="Multiplier for causal direction suppression")
    parser.add_argument("--transluce", action="store_true",
                        help="Evaluate on official Transluce MMLU-hint benchmark")
    parser.add_argument("--output_json", type=str, default="introspection_results.json",
                        help="Path to save output JSON metrics")
    return parser.parse_args()


def run_benchmark(args):
    print("=" * 80)
    print("J-SPACE INTROSPECTION & PRIVILEGED ACCESS BENCHMARK")
    print("=" * 80)

    # 1. Load Model
    model, ignition_layer = load_model(
        model_key=args.model,
        custom_repo_id=args.custom_repo_id,
        device=args.device,
    )
    target_layer = args.target_layer if args.target_layer is not None else ignition_layer
    layers, lm_head = get_layers_and_head(model)

    print(f"Targeting Layer {target_layer} for Global Workspace activations.")

    # 2. Load Tasks
    tasks = get_reasoning_benchmark(use_transluce=args.transluce)
    print(f"Evaluating across {len(tasks)} benchmark items...\n")

    icl_records: List[Dict[str, Any]] = []
    cont_records: List[Dict[str, Any]] = []

    for item in tqdm(tasks, desc="Running Benchmark"):
        prompt = item["task"]
        target_inter = item["target_intermediate"]
        entity_type = item["entity_type"]

        # --- A. Base Forward Pass ---
        with model.trace() as tracer:
            with tracer.invoke(prompt) as invoker:
                base_logits = lm_head.output[0, -1, :].save()

        base_tok = base_logits.argmax().item()
        base_output_str = model.tokenizer.decode([base_tok]).strip()

        # --- B. Baseline 1: External Observer ---
        obs_str, obs_tok = run_observer_baseline(model, lm_head, prompt, entity_type)
        obs_correct = (target_inter.lower() in obs_str.lower())

        # --- C. Baseline 2: Pure ICL Self-Reporting (Our Proposal) ---
        icl_str, icl_tok, _ = run_pure_icl_self_report(
            model, layers, lm_head, prompt, base_output_str, target_layer, entity_type
        )
        icl_correct = (target_inter.lower() in icl_str.lower())

        # Causal Interchange Intervention on ICL
        icl_causal, ablated_icl_out = causal_ablation_check(
            model, layers, lm_head, prompt, target_layer, icl_tok, base_tok, args.ablation_strength
        )

        icl_records.append({
            "task_id": item["id"],
            "task": prompt,
            "target_intermediate": target_inter,
            "base_output": base_output_str,
            "self_report": icl_str,
            "observer_report": obs_str,
            "self_correct": icl_correct,
            "observer_correct": obs_correct,
            "causally_grounded": icl_causal,
            "ablated_output": ablated_icl_out,
        })

        # --- D. Baseline 3: Continuous Embedding Injection (<v>) (Belinda Li / Transluce) ---
        cont_str, cont_tok = run_continuous_injection_baseline(
            model, layers, lm_head, prompt, target_layer, entity_type
        )
        cont_correct = (target_inter.lower() in cont_str.lower())

        cont_causal, ablated_cont_out = causal_ablation_check(
            model, layers, lm_head, prompt, target_layer, cont_tok, base_tok, args.ablation_strength
        )

        cont_records.append({
            "task_id": item["id"],
            "task": prompt,
            "target_intermediate": target_inter,
            "base_output": base_output_str,
            "self_report": cont_str,
            "observer_report": obs_str,
            "self_correct": cont_correct,
            "observer_correct": obs_correct,
            "causally_grounded": cont_causal,
            "ablated_output": ablated_cont_out,
        })

    # 3. Compute Metrics
    icl_metrics = compute_metrics(icl_records)
    cont_metrics = compute_metrics(cont_records)

    # 4. Print Summary Table
    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARISON: PURE ICL vs. CONTINUOUS INJECTION vs. OBSERVER")
    print("=" * 80)

    table_data = [
        ["Model", args.model, args.model],
        ["Target Ignition Layer", target_layer, target_layer],
        ["Paradigm", "Pure ICL (Our Proposal)", "Continuous Injection (<v>, Transluce)"],
        ["Verbalization Accuracy", f"{icl_metrics['self_accuracy']:.2%}", f"{cont_metrics['self_accuracy']:.2%}"],
        ["Observer Accuracy", f"{icl_metrics['observer_accuracy']:.2%}", f"{cont_metrics['observer_accuracy']:.2%}"],
        ["Privileged Access Differential (PAD)", f"{icl_metrics['pad']:+.2%}", f"{cont_metrics['pad']:+.2%}"],
        ["McNemar p-value", f"{icl_metrics['p_value']:.4f}", f"{cont_metrics['p_value']:.4f}"],
        ["Causal Faithfulness (IIA)", f"{icl_metrics['causal_faithfulness_iia']:.2%}", f"{cont_metrics['causal_faithfulness_iia']:.2%}"],
        ["Confabulation Rate", f"{icl_metrics['confabulation_rate']:.2%}", f"{cont_metrics['confabulation_rate']:.2%}"],
    ]
    print(tabulate(table_data, tablefmt="fancy_grid"))

    # 5. Export JSON
    payload = {
        "model": args.model,
        "target_layer": target_layer,
        "icl_metrics": icl_metrics,
        "continuous_injection_metrics": cont_metrics,
        "icl_individual_records": icl_records,
        "continuous_individual_records": cont_records,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved full evaluation metrics to: {args.output_json}")


if __name__ == "__main__":
    run_benchmark(parse_args())
