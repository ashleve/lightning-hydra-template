from typing import Any, Optional


def get_best_checkpoint_path(trainer: Any) -> Optional[str]:
    """Return the best checkpoint path, or None to test the current model weights."""
    checkpoint_callback = trainer.checkpoint_callback
    if checkpoint_callback is None:
        return None

    return checkpoint_callback.best_model_path or None
