"""Shared FP32 all-exit ONNX export, inference, and parity helpers."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


INPUT_NAME = "image"
OUTPUT_NAMES = ("exit1_logits", "exit2_logits", "exit3_logits")
OPSET_VERSION = 17


def model_footprint(path: Path) -> dict:
    """Count the ONNX graph and each referenced external tensor file once."""
    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    root = path.resolve().parent
    files = {}

    def tensors(graph):
        yield from graph.initializer
        for sparse in graph.sparse_initializer:
            yield sparse.values
            yield sparse.indices
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    yield attribute.t
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    yield from attribute.tensors
                elif attribute.type == onnx.AttributeProto.SPARSE_TENSOR:
                    yield attribute.sparse_tensor.values
                    yield attribute.sparse_tensor.indices
                elif attribute.type == onnx.AttributeProto.SPARSE_TENSORS:
                    for sparse in attribute.sparse_tensors:
                        yield sparse.values
                        yield sparse.indices
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    yield from tensors(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        yield from tensors(nested)

    for tensor in tensors(model.graph):
        if tensor.data_location != onnx.TensorProto.EXTERNAL and not tensor.external_data:
            continue
        metadata = {entry.key: entry.value for entry in tensor.external_data}
        location = metadata.get("location")
        if not location or Path(location).is_absolute():
            raise ValueError(f"External tensor {tensor.name!r} has no relative location")
        external = (root / location).resolve()
        if not external.is_relative_to(root) or not external.is_file():
            raise FileNotFoundError(f"External tensor file is missing or outside the model directory: {location}")
        files.setdefault(external, {"location": location, "size_bytes": external.stat().st_size})

    graph_bytes = path.stat().st_size
    external_files = sorted(files.values(), key=lambda item: item["location"])
    external_bytes = sum(item["size_bytes"] for item in external_files)
    total = graph_bytes + external_bytes
    return {"onnx_graph_size_bytes": graph_bytes,
            "external_data_size_bytes": external_bytes,
            "deployment_size_bytes": total,
            "deployment_size_mib": total / (1024 ** 2),
            "external_data_files": external_files}


def export_all_exits(model: torch.nn.Module, path: Path, image_size: int) -> None:
    """Export one fixed-shape graph; forward() returns all three shared-weight exits."""
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
    model = model.cpu().float().eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, (torch.zeros(1, 3, image_size, image_size),), str(path),
                          input_names=[INPUT_NAME], output_names=list(OUTPUT_NAMES),
                          opset_version=OPSET_VERSION, export_params=True,
                          do_constant_folding=True)
    onnx.checker.check_model(str(path))


def load_session(path: Path, session_options: ort.SessionOptions | None = None) -> ort.InferenceSession:
    """Check graph structure and load an explicit CPU execution provider."""
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), sess_options=session_options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("Expected CPUExecutionProvider only")
    if [value.name for value in session.get_inputs()] != [INPUT_NAME]:
        raise ValueError("Expected one ONNX image input")
    if [value.name for value in session.get_outputs()] != list(OUTPUT_NAMES):
        raise ValueError("Expected three named exit-logit outputs")
    return session


def run_all_exits(session: ort.InferenceSession, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if images.ndim != 4 or images.shape[0] != 1 or images.shape[1] != 3:
        raise ValueError("The exported ONNX graph requires one RGB image")
    values = session.run(list(OUTPUT_NAMES), {INPUT_NAME: images.detach().cpu().numpy().astype(np.float32)})
    outputs = tuple(torch.from_numpy(np.asarray(value, dtype=np.float32)) for value in values)
    if any(output.shape != (1, 5) for output in outputs):
        raise ValueError("ONNX exits must each produce [1, 5] logits")
    return outputs


def compare_parity(model: torch.nn.Module, session: ort.InferenceSession, dataset,
                   samples: int, atol: float, rtol: float) -> dict:
    """Compare every exit on the first validation images in manifest order."""
    if samples < 1 or samples > len(dataset) or atol < 0 or rtol < 0:
        raise ValueError("Invalid parity sample count or tolerance")
    model = model.cpu().float().eval()
    differences = [[], [], []]
    reference = [[], [], []]
    candidate = [[], [], []]
    with torch.no_grad():
        for index in range(samples):
            image, _ = dataset[index]
            image = image.unsqueeze(0)
            pytorch_outputs = model(image)
            onnx_outputs = run_all_exits(session, image)
            for exit_index, (pytorch_output, onnx_output) in enumerate(zip(pytorch_outputs, onnx_outputs)):
                expected = pytorch_output.detach().cpu().numpy()
                actual = onnx_output.numpy()
                differences[exit_index].append(np.abs(expected - actual))
                reference[exit_index].append(expected)
                candidate[exit_index].append(actual)
    exits = {}
    for index, name in enumerate(OUTPUT_NAMES):
        errors = np.concatenate(differences[index], axis=0)
        expected = np.concatenate(reference[index], axis=0)
        actual = np.concatenate(candidate[index], axis=0)
        exits[name] = {"max_absolute_logit_difference": float(errors.max()),
                       "mean_absolute_logit_difference": float(errors.mean()),
                       "passed": bool(np.allclose(expected, actual, atol=atol, rtol=rtol))}
    return {"split": "validation", "samples": samples, "atol": atol, "rtol": rtol,
            "per_exit": exits, "passed": all(row["passed"] for row in exits.values())}
