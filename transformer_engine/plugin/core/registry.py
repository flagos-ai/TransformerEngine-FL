# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .types import OpImpl


@dataclass
class OpRegistrySnapshot:
    """Immutable snapshot of operator registry state"""

    impls_by_op: Dict[str, List[OpImpl]]
    generation: int


class OpRegistry:
    """
    Thread-safe registry for operator implementations.

    This registry stores operator implementations indexed by op_name and impl_id.
    Each operator can have multiple implementations from different backends/vendors.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Structure: {op_name: {impl_id: OpImpl}}
        self._impls_by_op: Dict[str, Dict[str, OpImpl]] = {}
        self._generation = 0

    def register_impl(self, impl: OpImpl) -> None:
        """
        Register a single operator implementation.

        Args:
            impl: OpImpl instance to register

        Raises:
            ValueError: If impl_id is already registered for this op_name
        """
        with self._lock:
            by_id = self._impls_by_op.setdefault(impl.op_name, {})
            if impl.impl_id in by_id:
                raise ValueError(
                    f"Duplicate impl_id '{impl.impl_id}' for op='{impl.op_name}'. "
                    f"Existing: {by_id[impl.impl_id]}, New: {impl}"
                )
            by_id[impl.impl_id] = impl
            self._generation += 1

    def register_many(self, impls: Sequence[OpImpl]) -> None:
        """
        Register multiple operator implementations.

        Args:
            impls: Sequence of OpImpl instances to register
        """
        with self._lock:
            seen = set()
            for impl in impls:
                key = (impl.op_name, impl.impl_id)
                if key in seen:
                    raise ValueError(
                        f"Duplicate impl_id '{impl.impl_id}' for op='{impl.op_name}' in batch"
                    )
                seen.add(key)

                existing = self._impls_by_op.get(impl.op_name, {})
                if impl.impl_id in existing:
                    raise ValueError(
                        f"Duplicate impl_id '{impl.impl_id}' for op='{impl.op_name}'. "
                        f"Existing: {existing[impl.impl_id]}, New: {impl}"
                    )

            for impl in impls:
                self._impls_by_op.setdefault(impl.op_name, {})[impl.impl_id] = impl
            if impls:
                self._generation += 1

    @property
    def generation(self) -> int:
        """Return the current registry generation.

        Deliberately lock-free: a plain int read is atomic, and callers that
        need the generation to agree with a candidate list use snapshot(),
        which reads both under the lock.

        Staying lock-free also keeps this readable by Dynamo, which matters
        because the Triton dispatch entries read it on every call. Dynamo
        cannot trace an RLock context manager: under the default
        torch.compile it splits the graph at each dispatched call (both sides
        are still captured), and under fullgraph=True it raises outright.
        """
        return self._generation

    def snapshot(self) -> OpRegistrySnapshot:
        """
        Create an immutable snapshot of current registry state.

        Returns:
            OpRegistrySnapshot with all registered implementations
        """
        with self._lock:
            impls_by_op = {op: list(by_id.values()) for op, by_id in self._impls_by_op.items()}
            generation = self._generation
        return OpRegistrySnapshot(impls_by_op=impls_by_op, generation=generation)

    def get_implementations(self, op_name: str) -> List[OpImpl]:
        """
        Get all implementations for a specific operator.

        Args:
            op_name: Name of the operator

        Returns:
            List of OpImpl for the operator (empty if not found)
        """
        with self._lock:
            by_id = self._impls_by_op.get(op_name, {})
            return list(by_id.values())

    def get_implementation(self, op_name: str, impl_id: str) -> OpImpl | None:
        """
        Get a specific implementation by op_name and impl_id.

        Args:
            op_name: Name of the operator
            impl_id: Implementation ID

        Returns:
            OpImpl if found, None otherwise
        """
        with self._lock:
            by_id = self._impls_by_op.get(op_name, {})
            return by_id.get(impl_id)

    def list_operators(self) -> List[str]:
        """
        List all registered operator names.

        Returns:
            List of operator names
        """
        with self._lock:
            return list(self._impls_by_op.keys())

    def clear(self) -> None:
        """Clear all registered implementations"""
        with self._lock:
            self._impls_by_op.clear()
            self._generation += 1
