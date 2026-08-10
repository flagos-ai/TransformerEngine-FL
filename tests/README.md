# TransformerEngine-FL Test Suite

## Quick Start

```bash
# Run tests
bash qa/<test_type>/test.sh
```

## Directory Structure

```
tests/
├── cpp/                                  # C++ core functionality tests
│   ├── operator/                         # C++ operator layer tests (basic/core operator validation)
│   └── util/                             # C++ utility function tests (common helper unit tests)
├── cpp_distributed/                      # C++ distributed functionality tests (communication/parallelism)
├── jax/                                  # JAX framework adaptation tests (JAX backend validation)
└── pytorch/                              # Full PyTorch framework tests
    ├── attention/                        # PyTorch attention mechanism tests (FlashAttention/MLA etc.)
    ├── debug/                            # Debug-specific tests (issue reproduction/debug tooling)
    │   └── test_configs/                 # Debug test configurations (params/cases for different scenarios)
    ├── distributed/                      # PyTorch distributed tests (DDP/FSDP/communication)
    ├── nvfp4/                            # NVFP4 quantization tests (NVIDIA FP4 operator/inference)
    └── references/                       # Reference implementation tests (consistency vs baseline)
```

## Adding Tests

### Unit Test
Add test file: 
- `tests/cpp/test_<name>.cpp` & `tests/cpp/CMakeLists.txt`
- `tests/cpp_distributed/test_<name>.py` & `tests/cpp_distributed/CMakeLists.txt`
- `tests/jax/test_<name>.py`
- `tests/pytorch/test_<name>.py`
