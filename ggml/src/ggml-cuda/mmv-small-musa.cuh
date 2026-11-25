#pragma once
#include "common.cuh"

// MTT S80: matrix-vector product for matrices with few rows (e.g. the 48-row bf16 DeltaNet alpha/beta projections).
bool ggml_cuda_musa_mmv_small(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
