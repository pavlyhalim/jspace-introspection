"""
Publication-Quality Visualization & LaTeX Table Generator
Author: Pavly Halim <pavlyhalim@gmail.com>
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


def generate_plots_and_tables(json_path: str, output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path) as f:
        data = json.load(f)

    icl = data["icl_metrics"]
    cont = data["continuous_injection_metrics"]
    model_name = data.get("model", "Model")

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Plot 1: Privileged Access Differential ---
    categories = [
        "Pure ICL\nSelf-Report",
        "Observer\n(ICL Baseline)",
        "Continuous <v>\nInjection",
        "Observer\n(<v> Baseline)",
    ]
    accuracies = [
        icl["self_accuracy"] * 100,
        icl["observer_accuracy"] * 100,
        cont["self_accuracy"] * 100,
        cont["observer_accuracy"] * 100,
    ]
    colors = ["#2b5c8f", "#7ea4cc", "#d95f02", "#fdae6b"]

    bars = ax1.bar(categories, accuracies, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("Accuracy on Intermediate Concept (%)", fontweight="bold")
    ax1.set_title(f"Privileged Access Differential (PAD)\n[{model_name}]", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # --- Plot 2: Causal Faithfulness vs Confabulation ---
    groups = ["Pure ICL", "Continuous <v> Injection"]
    faithfulness = [icl["causal_faithfulness_iia"] * 100, cont["causal_faithfulness_iia"] * 100]
    confabulation = [icl["confabulation_rate"] * 100, cont["confabulation_rate"] * 100]

    x = range(len(groups))
    width = 0.35

    ax2.bar([p - width/2 for p in x], faithfulness, width=width, label="Causally Grounded (IIA)", color="#2ca02c", edgecolor="black")
    ax2.bar([p + width/2 for p in x], confabulation, width=width, label="Post-Hoc Confabulation", color="#d62728", edgecolor="black")

    ax2.set_ylabel("Percentage of Explanations (%)", fontweight="bold")
    ax2.set_title(f"Causal Grounding under Interchange Intervention\n[{model_name}]", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "jspace_vs_continuous_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved publication comparison plot to: {plot_path}")

    # --- LaTeX Table Generator ---
    latex_table = f"""
\\begin{{table}}[t]
\\centering
\\small
\\caption{{Empirical comparison between Pure In-Context Learning (J-Space) and Continuous Embedding Injection (\\citet{{li2025training}}) on {model_name}.}}
\\label{{tab:jspace_comparison}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Paradigm}} & \\textbf{{Self-Acc (\\%)}} & \\textbf{{Observer (\\%)}} & \\textbf{{$\\Delta_{{\\text{{priv}}}}$ (\\%)}} & \\textbf{{Causal IIA (\\%)}}\\\\
\\midrule
Pure ICL (Our Proposal) & {icl['self_accuracy']*100:.1f} & {icl['observer_accuracy']*100:.1f} & {icl['pad']*100:+.1f} & {icl['causal_faithfulness_iia']*100:.1f}\\\\
Continuous $\\langle v\\rangle$ (Transluce) & {cont['self_accuracy']*100:.1f} & {cont['observer_accuracy']*100:.1f} & {cont['pad']*100:+.1f} & {cont['causal_faithfulness_iia']*100:.1f}\\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    table_path = os.path.join(output_dir, "results_table.tex")
    with open(table_path, "w") as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to: {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="introspection_results.json")
    parser.add_argument("--outdir", type=str, default="figures")
    args = parser.parse_args()
    generate_plots_and_tables(args.json, args.outdir)
