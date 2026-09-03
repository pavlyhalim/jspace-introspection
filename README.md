# J-Space Introspection: Benchmarking Activation Verbalization

This repository evaluates whether language models can faithfully verbalize their internal working memory representations without specialized training.

We benchmark three paradigms on the official Transluce MMLU-hint dataset (Li et al., arXiv:2511.08579):
1. **Pure In-Context Learning (J-Space Self-Report):** Evaluating native introspection from the model's Global Workspace without weights or embedding modification.
2. **Continuous Embedding Injection ($\langle v\rangle$):** The linear projection paradigm of Li et al. (Transluce AI), which injects intermediate activations into Layer 0 token embeddings.
3. **External Observer Model:** An identical model evaluated on input text alone, establishing the **Privileged Access Differential (PAD)** ($\Delta_{\text{priv}} = \text{Acc}_{\text{self}} - \text{Acc}_{\text{observer}}$).

Every reported concept is evaluated for **Causal Faithfulness** via interchange interventions on the residual stream at workspace ignition depth ($\ell \approx 0.65 L$).

---

## Empirical Benchmark Results

Evaluated on NVIDIA A100 GPU across 104 items from Transluce's MMLU-hint benchmark:

| Metric | `meta-llama/Llama-3.1-8B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` | `google/gemma-4-E2B-it` |
| :--- | :---: | :---: | :---: |
| **Total Layers / Target Depth** | 32 / Layer 18 | 28 / Layer 18 | 35 / Layer 18 |
| **Pure ICL Explanation Accuracy** | **84.6%** | **90.4%** | **16.3%** |
| **Observer Baseline Accuracy** | 72.1% | 76.9% | 0.0% |
| **Privileged Access Differential ($\Delta_{\text{priv}}$)** | **+12.5%** ($p = 0.0072$) | **+13.5%** ($p = 0.0005$) | **+16.3%** ($p < 0.0001$) |
| **Continuous Injection ($\langle v\rangle$) Accuracy** | 0.0% | 0.0% | 0.0% |
| **Causal Faithfulness (IIA)** | **1.0%** | **1.0%** | **18.3%** |
| **Post-Hoc Confabulation Rate** | **99.0%** | **99.0%** | **81.7%** |

---

## Repository Structure

```
jspace-introspection/
├── .gitignore                      # Excludes .env, .DS_Store, __pycache__, checkpoints
├── .env.example                    # Template for Hugging Face authentication
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation & benchmark summary
├── run_experiments.py              # CLI benchmark runner with --transluce support
├── visualize.py                    # Publication chart and LaTeX table generator
├── jspace/
│   ├── models.py                   # Multi-model loader with automatic .env token support
│   ├── benchmark.py                # Multi-step arithmetic, relational, and hint tasks
│   ├── metrics.py                  # PAD, McNemar significance, and IIA faithfulness
│   ├── interventions.py            # nnsight residual stream projection engine
│   └── baselines/
│       ├── observer.py             # Input-only external observer
│       ├── pure_icl.py             # Two-turn ICL self-report
│       └── continuous_injection.py # Transluce <v> embedding projection
├── notebooks/
│   ├── run_benchmark.ipynb         # Turnkey starter notebook for Google Colab / A100
│   ├── llama3.1_8b_results.ipynb   # Executed evaluation notebook for LLaMA-3.1-8B
│   ├── qwen2.5_7b_results.ipynb    # Executed evaluation notebook for Qwen-2.5-7B
│   └── gemma4_results.ipynb        # Executed evaluation notebook for Gemma-4-E2B
└── figures/
    ├── llama_benchmark_results.png
    ├── qwen_benchmark_results.png
    └── gemma4_benchmark_results.png
```

---

## Setup & Execution

### 1. Environment Configuration
Copy the environment template and configure your Hugging Face token (needed for gated models like LLaMA-3.1):
```bash
cp .env.example .env
# Edit .env: HF_TOKEN=your_huggingface_token_here
```

### 2. Dependencies
```bash
pip install -r requirements.txt
```

### 3. Running the Benchmark
On an A100 or CUDA GPU:
```bash
# LLaMA-3.1-8B (evaluates on Transluce MMLU-hint benchmark):
python run_experiments.py --model llama-3.1-8b --transluce --device cuda --output_json llama_results.json

# Qwen-2.5-7B (ungated, no HF token needed):
python run_experiments.py --model qwen-2.5-7b --transluce --device cuda --output_json qwen_results.json

# Gemma-4:
python run_experiments.py --model gemma-4-e2b --transluce --device cuda --output_json gemma_results.json
```

For local verification on CPU or Apple Silicon (MPS):
```bash
python run_experiments.py --model qwen-2.5-0.5b --device cpu
```

### 4. Visualizations & Tables
```bash
python visualize.py --json llama_results.json --outdir figures
```

---

## Metrics

1. **Privileged Access Differential (PAD):**
   $$\Delta_{\text{priv}} = \text{Accuracy}_{\text{self}} - \text{Accuracy}_{\text{observer}}$$
   Measures whether accessing internal residual states provides an explanatory advantage over an external observer given identical text.

2. **Causal Faithfulness (Interchange Intervention Accuracy, IIA):**
   Percentage of cases where projecting the reported token direction out of the residual stream at Layer $\ell$ alters the model's output:
   $$h_\ell \gets h_\ell - (h_\ell \cdot \hat{d})\hat{d}$$

3. **Confabulation Rate:**
   $$\text{Confabulation} = 1.0 - \text{IIA}$$
   Quantifies how often a model emits a plausible verbal report that has zero causal influence over its output token.
