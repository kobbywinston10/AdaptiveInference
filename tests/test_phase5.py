import json
import math

import pytest
import torch
from torch.nn import functional as F

from src.models.adaptive import AdaptiveResNet50
from src.routing.calibration import binary_ece, fit_temperature
from src.routing.policy import RoutingPolicy, binary_entropy, load_policy, route_all, route_one, select_thresholds, uncertainty


def test_binary_entropy_and_aggregation():
    probabilities = torch.tensor([[0.0, 1.0, 0.5, 0.1, 0.9]])
    entropy = binary_entropy(probabilities)
    assert torch.isfinite(entropy).all()
    assert entropy[0, 0] < 0.001
    assert entropy[0, 2] == pytest.approx(math.log(2), rel=1e-6)
    logits = torch.tensor([[0., 8., 8., 8., 8.]])
    assert uncertainty(logits, aggregation="max") == pytest.approx(math.log(2), rel=1e-6)
    assert uncertainty(logits, aggregation="mean") < uncertainty(logits, aggregation="max")


def test_temperature_fit_and_ece():
    logits = torch.tensor([[5., 5., 5., 5., 5.], [-5., -5., -5., -5., -5.], [2., 2., 2., 2., 2.], [-2., -2., -2., -2., -2.]])
    labels = torch.tensor([[1., 1., 0., 1., 0.], [0., 0., 1., 0., 1.], [1., 0., 1., 0., 1.], [0., 1., 0., 1., 0.]])
    temperature = fit_temperature(logits, labels)
    assert 0.05 <= temperature <= 20
    assert F.binary_cross_entropy_with_logits(logits / temperature, labels) <= F.binary_cross_entropy_with_logits(logits, labels) + 1e-7
    assert 0 <= binary_ece(torch.sigmoid(logits), labels) <= 1
    with pytest.raises(ValueError):
        fit_temperature(logits, labels[:, :4])


def test_validation_threshold_selection_and_policy_load(tmp_path):
    labels = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]])
    final = torch.where(labels.bool(), torch.full_like(labels, 4.), torch.full_like(labels, -4.))
    first = final.clone()
    first[1] = 0
    second = final.clone()
    exits = (first, second, final)
    policy, selected, sweep = select_thresholds(exits, labels, (1., 1., 1.), quantile_steps=4, max_bce_increase=0.1)
    assert sweep and selected["validation_bce"] <= selected["final_exit_validation_bce"] + 0.1 + 1e-8
    routed, indices = route_all(exits, policy)
    assert routed.shape == labels.shape and indices.shape == (4,)
    assert (indices >= 1).all() and (indices <= 3).all()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"synthetic checkpoint fingerprint")
    from src.utils.run import file_sha256
    artifact = tmp_path / "calibration.json"
    artifact.write_text(json.dumps({"schema_version": 1, "checkpoint_sha256": file_sha256(checkpoint), "policies": {"calibrated": {"policy": policy.to_dict()}}}))
    assert load_policy(artifact, "calibrated", checkpoint) == policy
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        load_policy(artifact, "calibrated", checkpoint)


def test_real_conditional_routing_skips_layers():
    torch.set_num_threads(2)
    model = AdaptiveResNet50().eval()
    image = torch.randn(1, 3, 32, 32)
    calls = [0, 0]
    hooks = [
        model.backbone.layer3.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1)),
        model.backbone.layer4.register_forward_hook(lambda *_: calls.__setitem__(1, calls[1] + 1)),
    ]
    try:
        policy = RoutingPolicy(math.log(2), -1)
        probabilities, index = route_one(model, image, policy)
        assert index == 1 and probabilities.shape == (1, 5)
        assert calls == [0, 0]
    finally:
        for hook in hooks:
            hook.remove()
