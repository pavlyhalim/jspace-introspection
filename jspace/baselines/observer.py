from typing import Tuple, Any


def run_observer_baseline(
    model: Any,
    lm_head: Any,
    task_prompt: str,
    entity_type: str,
) -> Tuple[str, int]:
    """Prompts the model as an external observer with access to input text only."""
    observer_prompt = (
        f"A language model completed this task: '{task_prompt}'\n"
        f"Predict the intermediate {entity_type} that was active in its internal workspace.\n"
        f"Answer:"
    )

    with model.trace() as tracer:
        with tracer.invoke(observer_prompt) as invoker:
            obs_logits = lm_head.output[0, -1, :] if lm_head.output.dim() == 3 else lm_head.output[-1, :]
            saved_logits = obs_logits.save()

    obs_tok = saved_logits.argmax().item()
    obs_str = model.tokenizer.decode([obs_tok]).strip()
    return obs_str, obs_tok
