#pragma once
#include "common.cuh"

// MTT S80 split-KV attention for small query batches; false = not handled (use the ggml kernels)
bool ggml_cuda_flash_attn_ext_musa_ok(const ggml_tensor * dst);
bool ggml_cuda_flash_attn_ext_musa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
