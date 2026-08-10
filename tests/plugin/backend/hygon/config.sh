#!/usr/bin/env bash

# Hygon/DTK workflow configuration. Keep backend-specific selection policy
# here, and keep the runner focused on executing explicitly supported tests.

HYGON_ONNX_SKIP_GROUPS=(
    "test_export_linear"
    "test_export_layernorm_linear"
    "test_export_layernorm_mlp"
    "test_export_core_attention"
    "test_export_transformer_layer"
    "test_export_multihead_attention"
    "test_export_gpt_generation"
)
