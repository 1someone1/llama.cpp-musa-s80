#pragma once
#include "common.cuh"

// MTT S80: graph-level fusions for models with long chains of tiny element-wise ops (Inkling 2 Mini).
// Returns the number of extra nodes consumed (0 = no match), like ggml_cuda_try_fuse.
int ggml_cuda_musa_try_fuse(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);

// same-shape f32/f16 copies (CPY/CONT/DUP) and f32 concat without per-element index divisions
bool ggml_cuda_musa_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_musa_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// tiny Q8_0 matmuls (<= 128 rows or K <= 256), f32 activations, N <= 4; eligible tensors stay out of the tensor-core path
bool ggml_cuda_musa_q8_small_eligible(const ggml_tensor * src0);
bool ggml_cuda_musa_q8_small(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// consecutive MUL_MATs sharing src1 (Q8_0 tensor-core weights in one launch), in mul-mat-q4-mma.cu
int ggml_cuda_musa_mm_group(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);

// single-token attention, f16 K/V, D = 128, per-head masks and sinks allowed (Inkling decode); false = not handled
bool ggml_cuda_musa_fa_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// debug: skip a fusion's kernels by name (GGML_MUSA_SKIP_FUSED)
bool musa_skip_fused(const char * name);

// attention for 2-4 query tokens (MTP verify), D = 128, f16 K/V, per-head masks and sinks; false = not handled
bool ggml_cuda_musa_fa_small(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
