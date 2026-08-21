from types import SimpleNamespace
import warnings

import pytest
import torch

from transformer_engine.plugin.core.backends.vendor.compat import VendorTECompat, _WARNED_OPERATIONS


@pytest.fixture(autouse=True)
def clear_compat_warnings():
    _WARNED_OPERATIONS.clear()


def test_current_vendor_abi_is_preferred_without_warning():
    calls = []
    extension = SimpleNamespace(
        fused_score_for_moe_aux_loss_bwd=lambda *args: calls.append(args) or "current"
    )
    compat = VendorTECompat(extension, "TestVendor")
    args = (object(), object(), torch.empty(3, 5), 2, "softmax")

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        assert compat.fused_score_for_moe_aux_loss_bwd(*args) == "current"

    assert not warnings_record
    assert calls == [args]


def test_removed_router_dimensions_are_restored_for_legacy_abi():
    calls = []

    def legacy(*args):
        calls.append(args)
        if len(args) != 7:
            raise TypeError("incompatible function arguments")
        return "legacy"

    compat = VendorTECompat(SimpleNamespace(fused_score_for_moe_aux_loss_bwd=legacy), "TestVendor")
    args = (object(), object(), torch.empty(3, 5), 2, "softmax")

    with pytest.warns(RuntimeWarning, match="being upgraded"):
        assert compat.fused_score_for_moe_aux_loss_bwd(*args) == "legacy"

    assert calls[1][:2] == (3, 5)
    assert calls[1][2:] == args


@pytest.mark.parametrize("operation", ["group_quantize", "bgrad_group_quantize"])
@pytest.mark.parametrize(("vendor", "legacy_arg_count"), [("Hygon", 2), ("MUSA", 4)])
def test_group_quantize_uses_vendor_specific_legacy_abi(operation, vendor, legacy_arg_count):
    calls = []

    def legacy(*args):
        calls.append(args)
        if len(args) != legacy_arg_count:
            raise TypeError("incompatible function arguments")
        return "legacy"

    compat = VendorTECompat(SimpleNamespace(**{operation: legacy}), vendor)
    with pytest.warns(RuntimeWarning, match="being upgraded"):
        assert getattr(compat, operation)(object(), object(), 4, [1, 2, 3, 4], None) == "legacy"

    assert [len(args) for args in calls] == [5, legacy_arg_count]


def test_non_equivalent_new_semantics_are_not_silently_dropped():
    def legacy(*args):
        raise TypeError("incompatible function arguments")

    compat = VendorTECompat(SimpleNamespace(clamped_swiglu=legacy), "TestVendor")
    with pytest.warns(RuntimeWarning, match="being upgraded"), pytest.raises(TypeError):
        compat.clamped_swiglu(object(), object(), 7.0, 1.702, 0.5)


def test_generic_gemm_supports_enflame_legacy_abi():
    calls = []

    def legacy(*args):
        calls.append(args)
        if len(args) != 17:
            raise TypeError("incompatible function arguments")
        return "legacy"

    compat = VendorTECompat(SimpleNamespace(generic_gemm=legacy), "Enflame")
    current_args = tuple(range(22))
    with pytest.warns(RuntimeWarning, match="being upgraded"):
        assert compat.generic_gemm(*current_args) == "legacy"

    assert calls[1] == (*current_args[:12], current_args[14], *current_args[18:])
