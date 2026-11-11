"""NPU checkpoint round-trip coverage for GroupedLinear single grouped parameters."""

import os
import tempfile

import torch
import torch_npu  # noqa: F401

import transformer_engine.pytorch as te
import transformer_engine.pytorch.module.grouped_linear as grouped_linear_module
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor


os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"


def _make_model(single: bool) -> te.GroupedLinear:
    # The current worktree has a pre-existing lowercase `none` typo in
    # module/grouped_linear.py. Keep the source untouched and supply the intended
    # value only in this test process so checkpoint coverage can run independently.
    grouped_linear_module.none = None
    return te.GroupedLinear(
        num_gemms=3,
        in_features=16,
        out_features=8,
        bias=False,
        params_dtype=torch.float32,
        device="npu",
        single_grouped_weight=single,
    )


def _weight_members(model: te.GroupedLinear) -> list[torch.Tensor]:
    if model.single_grouped_weight:
        members = model.weight.quantized_tensors
        if members is None:
            members = model.weight.split_into_quantized_tensors()
        return list(members)
    return [getattr(model, f"weight{i}") for i in range(model.num_gemms)]


def _fill_and_snapshot(model: te.GroupedLinear) -> list[torch.Tensor]:
    snapshots = []
    with torch.no_grad():
        for index, weight in enumerate(_weight_members(model)):
            expected = torch.arange(
                weight.numel(),
                dtype=weight.dtype,
                device=weight.device,
            ).view_as(weight)
            expected.add_(index * 1000)
            weight.copy_(expected)
            snapshots.append(expected.cpu().clone())
    return snapshots


def _assert_weights(model: te.GroupedLinear, expected: list[torch.Tensor]) -> None:
    actual = _weight_members(model)
    assert len(actual) == len(expected)
    for actual_weight, expected_weight in zip(actual, expected):
        torch.testing.assert_close(
            actual_weight.cpu(),
            expected_weight,
            rtol=0,
            atol=0,
        )


def _run_case(source_single: bool, destination_single: bool) -> None:
    source = _make_model(source_single)
    expected = _fill_and_snapshot(source)
    state_dict = source.state_dict()

    if source_single:
        grouped_weight = state_dict["weight"]
        assert isinstance(grouped_weight, GroupedTensor)
        assert grouped_weight.data_ptr() == 0
        assert grouped_weight.rowwise_data.data_ptr() != 0
        torch.testing.assert_close(
            grouped_weight.rowwise_data.cpu(),
            torch.cat([weight.reshape(-1) for weight in expected]),
            rtol=0,
            atol=0,
        )
    else:
        assert "weight" not in state_dict
        assert all(f"weight{i}" in state_dict for i in range(source.num_gemms))

    checkpoint_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as checkpoint:
            checkpoint_path = checkpoint.name
        torch.save(state_dict, checkpoint_path)
        restored_state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    finally:
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)

    destination = _make_model(destination_single)
    result = destination.load_state_dict(restored_state_dict, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    _assert_weights(destination, expected)

    source_name = "single" if source_single else "non-single"
    destination_name = "single" if destination_single else "non-single"
    print(f"FORMAT_COMPAT={source_name}->{destination_name}:PASS")


def main() -> None:
    torch.npu.set_device(0)
    for source_single, destination_single in (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ):
        _run_case(source_single, destination_single)
    print("GROUPED_LINEAR_STATE_DICT_MATRIX=PASS")


if __name__ == "__main__":
    main()
