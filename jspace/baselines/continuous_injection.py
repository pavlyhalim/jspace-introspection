"""
Baseline 3: Continuous Embedding Injection (<v>)
Implements Belinda Li / TransluceAI's exact paradigm from arXiv:2511.08579.
Projects intermediate residual stream activations through a linear projection Pi_l
and injects them into the Layer 0 token embedding space.
"""

from typing import Tuple, Any
import torch
import torch.nn as nn


class ContinuousEmbeddingProjector(nn.Module):
    """Linear projection matrix Pi_l matching Transluce embed_projs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def run_continuous_injection_baseline(
    model: Any,
    layers: Any,
    lm_head: Any,
    task_prompt: str,
    target_layer: int,
    entity_type: str,
    projector: ContinuousEmbeddingProjector | None = None,
) -> Tuple[str, int]:
    """
    1. Extracts activation vector v at target_layer.
    2. Projects v into token embedding space via Pi_l.
    3. Injects projected vector into Layer 0 inputs_embeds (<v> token).
    4. Explainer generates the verbalized feature description.
    """
    # Step A: Extract activation vector at target_layer
    with model.trace() as tracer:
        with tracer.invoke(task_prompt) as invoker:
            out_obj = layers[target_layer].output
            act_tensor = out_obj[0] if isinstance(out_obj, tuple) else out_obj
            if act_tensor.dim() == 3:
                act_v = act_tensor[:, -1, :].save()
            else:
                act_v = act_tensor[-1, :].save()

    v_vector = act_v.clone()
    if projector is None:
        v_norm = v_vector / (torch.norm(v_vector) + 1e-8)
    else:
        v_norm = projector(v_vector)

    # Step B: Inject into explainer prompt at Layer 0
    explainer_prompt = f"The intermediate internal representation for {entity_type} <v> corresponds to:"
    
    with model.trace() as tracer:
        with tracer.invoke(explainer_prompt) as invoker:
            in_obj = layers[0].input
            first_layer_in = in_obj[0] if isinstance(in_obj, tuple) else in_obj
            v_vec = v_norm.squeeze().to(first_layer_in.device)
            
            if first_layer_in.dim() == 3:
                first_layer_in[:, -1, :] += v_vec * 2.0
            elif first_layer_in.dim() == 2:
                first_layer_in[-1, :] += v_vec * 2.0
            
            gen_logits = lm_head.output[0, -1, :] if lm_head.output.dim() == 3 else lm_head.output[-1, :]
            saved_logits = gen_logits.save()

    gen_tok = saved_logits.argmax().item()
    gen_str = model.tokenizer.decode([gen_tok]).strip()
    return gen_str, gen_tok
