# API Audit Decision Rules

A symbol is required when upstream Python code calls it, the fork's plugin contract exposes it, or a supported backend needs it. A pybind export alone may be intentionally unsupported only with an explicit reason.

Check more than function names:

- parameter order, defaults, and keyword names;
- enum domains and conversion at the tex boundary;
- dataclass/NamedTuple fields such as AttentionParams;
- return tuple shape and optional values;
- registration priority and availability predicate;
- backend capability and fallback semantics.

Never use *args/**kwargs to conceal a signature mismatch unless the target CUDA implementation itself uses it and the exception is recorded.
