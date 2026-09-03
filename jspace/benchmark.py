import json
import urllib.request
from typing import List, Dict, Any


DEFAULT_REASONING_TASKS: List[Dict[str, Any]] = [
    {
        "id": "arith_01",
        "category": "multi_step_arithmetic",
        "task": "Calculate 17 * 23 + 4 in your head. Answer:",
        "target_intermediate": "391",
        "expected_final": "395",
        "entity_type": "intermediate numerical product",
    },
    {
        "id": "arith_02",
        "category": "multi_step_arithmetic",
        "task": "Calculate 14 * 18 - 12 in your head. Answer:",
        "target_intermediate": "252",
        "expected_final": "240",
        "entity_type": "intermediate numerical product",
    },
    {
        "id": "arith_03",
        "category": "multi_step_arithmetic",
        "task": "Calculate 25 * 16 / 4 in your head. Answer:",
        "target_intermediate": "400",
        "expected_final": "100",
        "entity_type": "intermediate numerical product",
    },
    {
        "id": "arith_04",
        "category": "multi_step_arithmetic",
        "task": "Calculate 13 * 19 + 7 in your head. Answer:",
        "target_intermediate": "247",
        "expected_final": "254",
        "entity_type": "intermediate numerical product",
    },
    {
        "id": "relation_01",
        "category": "relational_composition",
        "task": "What is the capital of the country containing the city of Milan? Answer:",
        "target_intermediate": "Italy",
        "expected_final": "Rome",
        "entity_type": "country",
    },
    {
        "id": "relation_02",
        "category": "relational_composition",
        "task": "What is the capital of the country containing the city of Barcelona? Answer:",
        "target_intermediate": "Spain",
        "expected_final": "Madrid",
        "entity_type": "country",
    },
    {
        "id": "relation_03",
        "category": "relational_composition",
        "task": "What is the capital of the country containing the city of Munich? Answer:",
        "target_intermediate": "Germany",
        "expected_final": "Berlin",
        "entity_type": "country",
    },
    {
        "id": "relation_04",
        "category": "relational_composition",
        "task": "What continent is the nation containing Tokyo located in? Answer:",
        "target_intermediate": "Japan",
        "expected_final": "Asia",
        "entity_type": "nation",
    },
]


def fetch_transluce_mmlu_hint_tasks(limit: int = 100) -> List[Dict[str, Any]]:
    """Loads official MMLU-hint attribution rows from Transluce's public Hugging Face dataset."""
    url = f"https://datasets-server.huggingface.co/rows?dataset=Transluce/input_ablation_llama_3.1_8b_instruct_mmlu_hint&config=default&split=test&offset=0&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tasks = []
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            for r in data.get("rows", []):
                row = r.get("row", {})
                q = row.get("question", "")
                hint = row.get("hint", "").replace("Hint: ", "").strip()
                prompt = f"Question: {q}\nChoices: {row.get('choices', [])}\nHint: {hint}\nAnswer:"
                tasks.append({
                    "id": f"mmlu_hint_{row.get('index', len(tasks))}",
                    "category": "hint_attribution",
                    "task": prompt,
                    "target_intermediate": hint,
                    "expected_final": row.get("random_hint_prediction", "").strip(),
                    "entity_type": "hint letter",
                })
    except Exception as e:
        print(f"[Benchmark] Notice: Remote dataset fetch skipped ({e}). Using local tasks.")
    return tasks


def get_reasoning_benchmark(use_transluce: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
    """Returns benchmark items, optionally including Transluce MMLU-hint benchmark."""
    if use_transluce:
        remote_tasks = fetch_transluce_mmlu_hint_tasks(limit=limit)
        if remote_tasks:
            return remote_tasks + DEFAULT_REASONING_TASKS
    return DEFAULT_REASONING_TASKS
