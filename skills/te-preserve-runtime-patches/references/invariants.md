# Runtime Patch Invariants

Look for plugin dispatch, TE_DEVICE_TYPE and device-agnostic tensor allocation, import-time patching and optional dependency guards, attention/tensor/module/optimizer argument forwarding, vendor-specific patches and fallback behavior, and runtime environment variables.

A textual merge can be clean while deleting a dispatch call or changing a keyword name. Compare call sites and behavior, not only conflict markers.
