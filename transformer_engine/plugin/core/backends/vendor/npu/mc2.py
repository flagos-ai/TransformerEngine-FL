"""Ascend MC2 capability checks and routing-map implementation pairing."""

from dataclasses import replace
from functools import partial

from ....manager import get_default_manager
from ....policy import get_policy, policy_context


def is_moe_fusion_available(group=None, device=None):
    """Probe locally first. All EP ranks must participate in the group probe."""
    manager = get_default_manager()
    for name in ("moe_alltoallv_grouped_matmul", "moe_grouped_matmul_alltoallv"):
        try:
            fn = manager.resolve(name)
        except RuntimeError:
            return False
        if group is not None:
            supports_group = getattr(fn, "_supports_group", None)
            if supports_group is None or not supports_group(group, device):
                return False
    return True


def supports_moe_counts(counts, hidden_size, element_size):
    """Check both collective directions from the shared [source, expert] CPU counts."""
    ep_size = counts.size(0)
    if hidden_size <= 0 or counts.max().item() >= 65536:
        return False
    # FC1 dispatch and FC2 combine exchange hidden-size rows in opposite directions.
    sent = counts.sum(dim=1)
    received = counts.reshape(ep_size, ep_size, -1).sum(dim=(0, 2))
    min_rows = min(sent.min().item(), received.min().item())
    max_rows = max(sent.max().item(), received.max().item())
    return min_rows * hidden_size * element_size >= 2 * 1024**2 and max_rows < 52428800


def get_moe_permutation_pair(tokens, routing_map, probs, num_out_tokens, topk):
    """Keep both endpoints on the implementation that owns the private indices."""
    manager = get_default_manager()
    names = ("moe_permute_with_routing_map", "moe_unpermute_with_routing_map")
    policy = get_policy()
    try:
        permute, unpermute = (manager.resolve_candidates(name)[0] for name in names)
    except RuntimeError:
        return None
    if (permute.kind, permute.vendor, permute.impl_id) != (
        unpermute.kind, unpermute.vendor, unpermute.impl_id
    ):
        return None
    supports = getattr(permute.fn, "_supports_inputs", None)
    if supports is None or not supports(tokens, routing_map, probs, num_out_tokens, topk):
        return None

    generation = manager.registry.generation
    sole_pair = all(len(manager.registry.get_implementations(name)) == 1 for name in names)
    order = dict(policy.per_op_order)
    order.update((name, (f"impl:{permute.impl_id}",)) for name in names)
    pinned = replace(policy, per_op_order=tuple(sorted(order.items())))

    def call(name, *args, **kwargs):
        # Preserve normal dispatch caching when there is only one registered pair.
        if sole_pair and get_policy() is policy and manager.registry.generation == generation:
            return manager.call(name, *args, **kwargs)
        with policy_context(pinned):
            return manager.call(name, *args, **kwargs)

    return partial(call, names[0]), partial(call, names[1])
