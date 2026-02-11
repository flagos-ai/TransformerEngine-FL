# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


pip3 install onnxruntime
pip3 install onnxruntime_extensions
pip3 install tensorrt

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/test_onnx_export.xml $TE_PATH/tests/pytorch/test_onnx_export.py -k "not (test_export_layernorm_mlp or test_export_layernorm_mlp_return_layernorm_output or test_export_layernorm_mlp_return_bias or test_export_layernorm_mlp_zero_centered_gamma or test_export_core_attention or test_export_multihead_attention_recipe or test_export_multihead_attention_no_input_layernorm or test_export_multihead_attention_cross_attn or test_export_multihead_attention_unfused_qkv_params or test_export_transformer_layer_recipe or test_export_transformer_layer_no_mask or test_export_transformer_layer_output_layernorm or test_export_transformer_layer_unfused_qkv_params or test_export_transformer_layer_zero_centered_gamma or test_export_transformer_layer_activation or test_export_gpt_generation or test_trt_integration)"
