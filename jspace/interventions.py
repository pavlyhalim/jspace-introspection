"""
Causal Interchange Interventions via nnsight
Executes surgical subspace projections on the residual stream at Global Workspace depth.
"""

from typing import Tuple, Any
import torch


def causal_ablation_check(
    model: Any,
    layers: Any,
    lm_head: Any,
    prompt: str,
    target_layer: int,
    reported_token_id: int,
    base_token_id: int,
    ablation_strength: float = 3.0,
) -> Tuple[bool, str]:
    """
    Projects out the direction corresponding to the reported token from target_layer residual stream.
    Returns: (is_causally_grounded, ablated_token_string)
    """
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            direction = lm_head.weight[reported_token_id]
            norm_dir = direction / (torch.norm(direction) + 1e-8)

            out_obj = layers[target_layer].output
            h_tensor = out_obj[0] if isinstance(out_obj, tuple) else out_obj

            if h_tensor.dim() == 3:
                proj = (h_tensor[:, -1, :] @ norm_dir).unsqueeze(-1) * norm_dir
                h_tensor[:, -1, :] -= proj * ablation_strength
            elif h_tensor.dim() == 2:
                proj = (h_tensor[-1, :] @ norm_dir) * norm_dir
                h_tensor[-1, :] -= proj * ablation_strength

            ablated_logits = lm_head.output[0, -1, :] if lm_head.output.dim() == 3 else lm_head.output[-1, :]
            saved_logits = ablated_logits.save()

    ablated_tok = saved_logits.argmax().item()
    ablated_str = model.tokenizer.decode([ablated_tok]).strip()

    is_causally_grounded = (ablated_tok != base_token_id)
    return is_causally_grounded, ablated_str
