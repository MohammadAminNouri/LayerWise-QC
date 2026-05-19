import torch
from torch import nn

from am_defect_detection.models import LateFusionEnsemble


class FixedModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.logits.repeat(x.shape[0], 1)


def test_late_fusion_probability_shape_and_normalization():
    model = LateFusionEnsemble(
        {
            "ot": FixedModel([3.0, 1.0, 0.0]),
            "mpm": FixedModel([0.0, 1.0, 3.0]),
        },
        weights={"ot": 0.5, "mpm": 0.5},
    )
    probs = model({"ot": torch.zeros(2, 3, 8, 8), "mpm": torch.zeros(2, 3, 8, 8)})
    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
