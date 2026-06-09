import pytest
from am_defect_detection.inference import load_model_bundle


def test_missing_checkpoint_errors_cleanly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_bundle(tmp_path / "missing.pt", modality="ot")
