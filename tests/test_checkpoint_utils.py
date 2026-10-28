from types import SimpleNamespace

from src.checkpoint_utils import get_best_checkpoint_path


def test_disabled_checkpointing_uses_current_model_weights() -> None:
    """A Trainer without checkpointing should test its current model weights."""
    trainer = SimpleNamespace(checkpoint_callback=None)

    assert get_best_checkpoint_path(trainer) is None


def test_missing_best_checkpoint_uses_current_model_weights() -> None:
    """An empty best-model path should fall back to current model weights."""
    trainer = SimpleNamespace(checkpoint_callback=SimpleNamespace(best_model_path=""))

    assert get_best_checkpoint_path(trainer) is None


def test_best_checkpoint_path_is_preserved() -> None:
    """A saved best-model path should be forwarded for testing."""
    trainer = SimpleNamespace(checkpoint_callback=SimpleNamespace(best_model_path="best.ckpt"))

    assert get_best_checkpoint_path(trainer) == "best.ckpt"
