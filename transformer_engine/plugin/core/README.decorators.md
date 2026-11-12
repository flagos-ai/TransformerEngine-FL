# Function and class overrides

Use `overridable` at a shared entry point and `override` on its backend
implementation. The decorators use the existing `OpRegistry`, `OpManager`,
vendor filters and selection policy. They do not introduce another registry.

```python
# Shared entry point: example_package/operations.py
from transformer_engine.plugin.core.decorators import overridable


@overridable
def operation(x, scale=1):
    return x * scale
```

```python
# Backend implementation
from transformer_engine.plugin.core.decorators import override, register_overrides


@override(
    "example_package.operations.operation",
    vendor="HYGON",
    supports=supports_operation,  # Receives the actual arguments, including kwargs.
)
def operation_hygon(x, scale=1):
    return backend_operation(x, scale)


def register_builtins(registry):
    # Keep the backend's existing availability checks before registration.
    register_overrides(registry, operation_hygon)
```

`supports_operation` and `backend_operation` in this example are supplied by
the backend. For methods, the predicate and implementation receive the same
`self` or `cls` argument as the original method.

## Selection and fallback

- The registry key is the original callable's `__module__ + "." + __qualname__`.
  Use that complete key in `override`; short names can collide across modules.
- `override` attaches registration metadata and returns the implementation
  unchanged. It does not initialize the manager. Register it through the
  backend's existing registration entry point.
- `vendor`, `kind`, `impl_id` and `priority` use the existing `OpImpl` semantics.
  A vendor implementation defaults to an ID such as `vendor.hygon`.
- `is_available()` is an optional dynamic availability check, following the
  existing `OpImpl` availability behavior. `supports(*args, **kwargs)` checks
  the actual device, dtype or shape before that candidate is called.
- Policy, availability and input compatibility are checked on each call.
  Selection is not permanently cached from the first tensor's device.
- If no candidate matches, the original runs. This uses
  `resolve_candidates(..., optional=True)`; existing manager callers retain
  their default error behavior when no implementation is available.
- Errors raised by a selected implementation or input predicate propagate.
  The decorators do not retry a different implementation after execution
  starts, including when `TE_FL_STRICT` would permit retries in `manager.call`.

## Classes and autograd

Ordinary decorated classes dispatch their constructor. The selected vendor
class is instantiated normally, including `__init__`. It can be a different
type: callers must not assume the result has the original class's exact type
or passes `isinstance(result, DecoratedClass)`. Subclasses of the decorated
class retain their own construction behavior.

For a `torch.autograd.Function`, selection happens at `.apply()` and chooses
the entire Function class. PyTorch binds the selected backward to that graph;
changing the plugin policy after forward does not select a different
backward. Override classes must also inherit `torch.autograd.Function`.
Direct calls to the decorated class's `.forward()` or `.backward()` are not
dispatched; normal autograd users should call `.apply()`.

Functions, instance methods, static/class methods, async functions, ordinary
constructors and autograd `.apply()` are covered by external validation.
Function metadata and introspection use `functools.wraps` / `__wrapped__`.
This is an opt-in mechanism: each shared entry point needs `@overridable`,
and each backend implementation must be registered and preserve its contract.
It does not patch all functions automatically. Arbitrary descriptors,
serialization, tracing/compilation and CUDA graph capture require validation
for the specific entry point before claiming support.

See [Hygon parallel CE](backends/vendor/hygon/README.cross_entropy.md) for a
complete autograd-class integration.
