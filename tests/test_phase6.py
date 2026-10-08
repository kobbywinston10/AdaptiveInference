import pytest
import torch

from src.models.adaptive import AdaptiveResNet50
from src.routing.budget import Budget, constrain_exit, evaluate_budget_from_all, route_one_budgeted
from src.routing.policy import RoutingPolicy


def test_budget_caps_preferred_vs_actual():
    early = torch.tensor([[8.] * 5, [0.] * 5, [0.] * 5])
    middle = torch.tensor([[0.] * 5, [8.] * 5, [0.] * 5])
    final = torch.tensor([[2.] * 5] * 3)
    labels = torch.zeros(3, 5)
    policy = RoutingPolicy(0.2, 0.2)
    expected = {
        Budget.LOW: ([1, 1, 1], 2 / 3),
        Budget.MEDIUM: ([1, 2, 2], 1 / 3),
        Budget.HIGH: ([1, 2, 3], 0),
    }
    for budget, (actual, forced_rate) in expected.items():
        result = evaluate_budget_from_all((early, middle, final), labels, policy, budget)
        assert result["preferred_exit"].tolist() == [1, 2, 3]
        assert result["actual_exit"].tolist() == actual
        assert result["budget_forced_exit_rate"] == pytest.approx(forced_rate)
    with pytest.raises(ValueError):
        constrain_exit(torch.tensor([1, 4]), "LOW")
    with pytest.raises(ValueError):
        constrain_exit(torch.tensor([1, 2]), "UNKNOWN")


@pytest.mark.parametrize("budget,expected_exit,expected_calls", [
    (Budget.LOW, 1, [0, 0]),
    (Budget.MEDIUM, 2, [1, 0]),
    (Budget.HIGH, 3, [1, 1]),
])
def test_real_budgeted_inference_skips_prohibited_layers(budget, expected_exit, expected_calls):
    torch.set_num_threads(2)
    model = AdaptiveResNet50().eval()
    calls = [0, 0]
    hooks = [
        model.backbone.layer3.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1)),
        model.backbone.layer4.register_forward_hook(lambda *_: calls.__setitem__(1, calls[1] + 1)),
    ]
    try:
        result = route_one_budgeted(model, torch.randn(1, 3, 32, 32), RoutingPolicy(-1, -1), budget)
        assert result["actual_exit"] == expected_exit
        assert calls == expected_calls
        assert result["probabilities"].shape == (1, 5)
        assert result["budget_forced"] is (budget != Budget.HIGH)
        if budget == Budget.HIGH:
            assert result["preferred_exit"] == 3
        else:
            assert result["preferred_exit"] is None
    finally:
        for hook in hooks:
            hook.remove()
