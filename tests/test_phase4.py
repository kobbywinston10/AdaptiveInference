from pathlib import Path

import pytest
import torch

from src.models.adaptive import AdaptiveResNet50, load_radimagenet_backbone
from src.training.distillation import (
    DistillationConfig,
    auxiliary_loss,
    bernoulli_kd,
    compare_kd,
    freeze_baseline,
    train_auxiliary_epoch,
)


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return AdaptiveResNet50().eval()


def test_output_shapes_and_real_checkpoint(model):
    path = Path(__file__).parents[1] / "weights" / "ResNet50.pt"
    if not path.is_file():
        pytest.skip("Optional local RadImageNet checkpoint is absent")
    load_radimagenet_backbone(model, str(path))
    with torch.no_grad():
        outputs = model.forward_all_exits(torch.randn(1, 3, 32, 32))
    assert len(outputs) == 3
    assert all(output.shape == (1, 5) for output in outputs)


@pytest.mark.parametrize("selected,expected", [(1, (0, 0)), (2, (1, 0)), (3, (1, 1))])
def test_conditional_skipping(model, selected, expected):
    calls = [0, 0]
    hooks = [
        model.backbone.layer3.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1)),
        model.backbone.layer4.register_forward_hook(lambda *_: calls.__setitem__(1, calls[1] + 1)),
    ]
    try:
        with torch.no_grad():
            logits, exit_index = model.forward_adaptive(
                torch.randn(1, 3, 32, 32), lambda index, _: index == selected
            )
        assert logits.shape == (1, 5)
        assert exit_index == selected
        assert tuple(calls) == expected
    finally:
        for hook in hooks:
            hook.remove()


def test_multilabel_kd_and_frozen_teacher():
    student = torch.zeros(2, 5, requires_grad=True)
    teacher = torch.ones(2, 5, requires_grad=True)
    loss = bernoulli_kd(student, teacher, 2.0)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None
    assert torch.isfinite(loss)

    cfg = DistillationConfig(kd_weight=0)
    labels = torch.tensor([[1., 0., 1., 0., 0.], [0., 1., 0., 1., 0.]])
    outputs = (torch.zeros(2, 5), torch.zeros(2, 5), torch.zeros(2, 5))
    assert torch.allclose(auxiliary_loss(outputs, labels, cfg), 2 * torch.log(torch.tensor(2.)))
    with pytest.raises(ValueError):
        DistillationConfig(temperature=0)


def test_freeze_baseline(model):
    freeze_baseline(model)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(not p.requires_grad for p in model.final.parameters())
    assert all(p.requires_grad for p in model.exit1.parameters())
    assert all(p.requires_grad for p in model.exit2.parameters())


def test_one_auxiliary_training_step_keeps_baseline_fixed(model):
    freeze_baseline(model)
    before_backbone = model.backbone.conv1.weight.detach().clone()
    before_final = model.final[2].weight.detach().clone()
    before_exit = model.exit1[2].weight.detach().clone()
    optimizer = torch.optim.SGD(
        list(model.exit1.parameters()) + list(model.exit2.parameters()), lr=0.01
    )
    batches = [(torch.randn(2, 3, 32, 32), torch.zeros(2, 5))]
    result = train_auxiliary_epoch(model, batches, optimizer, DistillationConfig())
    assert result > 0
    assert torch.equal(model.backbone.conv1.weight, before_backbone)
    assert torch.equal(model.final[2].weight, before_final)
    assert not torch.equal(model.exit1[2].weight, before_exit)


def test_kd_comparison_smoke(model):
    batches = [(torch.randn(2, 3, 32, 32), torch.zeros(2, 5))]
    results = compare_kd(model, batches, batches, 1, 0.001, DistillationConfig())
    assert set(results) == {"with_kd", "without_kd"}
    assert all(len(values) == 2 and all(v >= 0 for v in values) for values in results.values())
