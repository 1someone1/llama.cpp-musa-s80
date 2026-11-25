#pragma once
#include "common.cuh"

// MTT S80: launch a run of consecutive small f32->f32 GGML_OP_CPY nodes as one kernel.
// Returns the number of graph nodes consumed beyond node i (0 = not applicable, run node i normally).
int ggml_cuda_musa_try_batch_cpy(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
