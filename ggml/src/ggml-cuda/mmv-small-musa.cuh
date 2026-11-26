#pragma once
#include "common.cuh"

// MTT S80: matrix-vector product for matrices with few rows (e.g. the 48-row bf16 DeltaNet alpha/beta projections).
bool ggml_cuda_musa_mmv_small(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// MTT S80: small f32 x f32 products of activations (replaces muBLAS sgemm for them; graph-capturable)
bool ggml_cuda_musa_mm_f32_small(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// products with K <= this go to mm_f32_small at any size (also checked by the MUSA graph compatibility rule)
#ifndef MMF_SMALL_SHORT_K
#define MMF_SMALL_SHORT_K 64
#endif
