import numpy as np
from PIL import Image

from am_defect_detection.sensor_features import compute_sensor_descriptors


def test_sensor_features_are_finite(tmp_path):
    img = (np.random.default_rng(0).normal(128, 10, size=(32, 32)).clip(0, 255)).astype("uint8")
    p = tmp_path / "img.png"
    Image.fromarray(img).save(p)
    desc = compute_sensor_descriptors(p)
    assert desc
    assert all(np.isfinite(v) for v in desc.values() if isinstance(v, (int, float)))
