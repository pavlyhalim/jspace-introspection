"""
Baseline 2: Pure In-Context Learning (J-Space Self-Reporting)
Tests whether models can naturally verbalize internal activations via ICL alone without training.
"""

from typing import Tuple, Any


def run_pure_icl_self_report(
    model: Any,
    layers: Any,
    lm_head: Any,
    task_prompt: str,
    base_output_str: str,
    target_layer: int,
    entity_type: str,
) -> Tuple[str, int, Any]:
    """
    Executes a two-turn in-context introspection query without continuous embedding modification.
    Returns: (self_reported_string, self_reported_token_id, intermediate_activation_vector)
    """
    # 1. Capture the intermediate residual stream vector at target_layer
    with model.trace() as tracer:
        with tracer.invoke(task_prompt) as invoker:
            out_obj = layers[target_layer].output
            act_tensor = out_obj[0] if isinstance(out_obj, tuple) else out_obj
            if act_tensor.dim() == 3:
                act_vector = act_tensor[:, -1, :].save()
            else:
                act_vector = act_tensor[-1, :].save()

    # 2. Query the model in-context for its intermediate mental state
    self_prompt = (
        f"{task_prompt} {base_output_str}\n"
        f"[Self-Report Query]\n"
        f"What intermediate {entity_type} was active in your working memory before producing the final answer?\n"
        f"Answer:"
    )

    with model.trace() as tracer:
        with tracer.invoke(self_prompt) as invoker:
            gen_logits = lm_head.output[0, -1, :] if lm_head.output.dim() == 3 else lm_head.output[-1, :]
            saved_logits = gen_logits.save()

    self_tok = saved_logits.argmax().item()
    self_str = model.tokenizer.decode([self_tok]).strip()

    return self_str, self_tok, act_vector
