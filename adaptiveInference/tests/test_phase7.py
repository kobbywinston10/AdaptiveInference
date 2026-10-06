import torch
import pytest
from torch.utils.data import Dataset

from src.evaluation.degradation import DegradedDataset, average_flops, confident_wrong_labels, gaussian_blur


PARAMETERS = {
    "blur_sigma_pixels": {"mild": 1.0, "severe": 2.0},
    "noise_std_normalized": {"mild": 0.05, "severe": 0.15},
}


class SyntheticImages(Dataset):
    rows = [{"image": "synthetic_1.png"}, {"image": "synthetic_2.png"}]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        image = torch.zeros(3, 64, 64)
        image[:, 32, 32] = 1
        return image, torch.tensor([1., 0., 0., 0., 0.])


def test_blur_and_noise_are_evaluation_only_and_deterministic():
    base = SyntheticImages()
    original, labels = base[0]
    clean, clean_labels = DegradedDataset(base, "noise", "clean", PARAMETERS, 42)[0]
    assert torch.equal(clean, original) and torch.equal(clean_labels, labels)
    mild_blur, _ = DegradedDataset(base, "blur", "mild", PARAMETERS, 42)[0]
    severe_blur, _ = DegradedDataset(base, "blur", "severe", PARAMETERS, 42)[0]
    assert severe_blur[0, 32, 32] < mild_blur[0, 32, 32] < original[0, 32, 32]
    assert torch.equal(mild_blur[0], mild_blur[1])
    assert torch.equal(base[0][0], original)
    mild = DegradedDataset(base, "noise", "mild", PARAMETERS, 42)
    severe = DegradedDataset(base, "noise", "severe", PARAMETERS, 42)
    first, _ = mild[0]
    assert torch.equal(first, mild[0][0])
    assert torch.equal(first[0], first[2])
    assert not torch.equal(first, mild[1][0])
    assert (severe[0][0] - original).abs().mean() > (first - original).abs().mean()
    assert severe[0][0].min() >= -1 and severe[0][0].max() <= 1
    with pytest.raises(ValueError):
        DegradedDataset(base, "contrast", "mild", PARAMETERS, 42)


def test_confident_errors_and_optional_measured_flops():
    probabilities = torch.tensor([[0.95, 0.1, 0.1, 0.1, 0.1], [0.2, 0.8, 0.1, 0.1, 0.1]])
    labels = torch.tensor([[0., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]])
    flagged, wrong, confidence = confident_wrong_labels(probabilities, labels)
    assert flagged.tolist() == [True, False]
    assert wrong[0, 0] and confidence[0] == pytest.approx(0.95)
    assert average_flops(torch.tensor([1, 2, 3]), {"exit1": 1, "exit2": 2, "exit3": 3}) == pytest.approx(2)
    with pytest.raises(ValueError):
        average_flops(torch.tensor([1, 2]), {"exit1": 3, "exit2": 2, "exit3": 1})
