// MTT S80: few-row matrix x few-column matrix (y = W x, W [K, R] with R <= 512, x [K, N] with N <= 8).
// ggml's mmvf uses one 256-thread block per row with block-wide shuffle reductions (~60 us for a 48x5120 bf16
// matrix at N=4 on this GPU). Here 32 threads share a row and reduce through shared memory inside one
// 128-thread hardware warp (4 rows per block), so there is no __syncthreads and no emulated shuffle.
#include "mmv-small-musa.cuh"
#include "convert.cuh"

#define MMVS_MAXN 8

namespace {

template <typename T> __device__ __forceinline__ float mmvs_load(const T * p, int64_t i);
template <> __device__ __forceinline__ float mmvs_load<float>(const float * p, int64_t i) { return p[i]; }
template <> __device__ __forceinline__ float mmvs_load<half>(const half * p, int64_t i) { return __half2float(p[i]); }
template <> __device__ __forceinline__ float mmvs_load<nv_bfloat16>(const nv_bfloat16 * p, int64_t i) { return ggml_cuda_cast<float>(p[i]); }

template <typename T, int N>
__launch_bounds__(128, 4)
__global__ void mmvs_kernel(const T * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
                            const int K, const int R, const int64_t sw, const int64_t sx, const int64_t sy) {
    // one 128-thread hardware warp per row; tree reduction in shared memory with warp syncs only
    __shared__ float red[N][128];
    const int tid = threadIdx.x, r = blockIdx.x;
    float acc[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        acc[j] = 0.0f;
    }
    const T * wr = w + (int64_t) r*sw;
#pragma unroll 4
    for (int k = tid; k < K; k += 128) {
        const float wv = mmvs_load<T>(wr, k);
#pragma unroll
        for (int j = 0; j < N; ++j) {
            acc[j] = fmaf(wv, x[(int64_t) j*sx + k], acc[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
        red[j][tid] = acc[j];
    }
    __musa_syncwarp();
#pragma unroll
    for (int o = 64; o > 0; o >>= 1) {
        if (tid < o) {
#pragma unroll
            for (int j = 0; j < N; ++j) {
                red[j][tid] += red[j][tid + o];
            }
        }
        __musa_syncwarp();
    }
    if (tid < N) {
        y[(int64_t) tid*sy + r] = red[tid][0];
    }
}

template <typename T>
void mmvs_launch(cudaStream_t st, const T * w, const float * x, float * y, int K, int R, int N, int64_t sw, int64_t sx, int64_t sy) {
    const dim3 grid(R);
    switch (N) {
        case 1: mmvs_kernel<T, 1><<<grid, 128, 0, st>>>(w, x, y, K, R, sw, sx, sy); break;
        case 2: mmvs_kernel<T, 2><<<grid, 128, 0, st>>>(w, x, y, K, R, sw, sx, sy); break;
        case 3: mmvs_kernel<T, 3><<<grid, 128, 0, st>>>(w, x, y, K, R, sw, sx, sy); break;
        case 4: mmvs_kernel<T, 4><<<grid, 128, 0, st>>>(w, x, y, K, R, sw, sx, sy); break;
        case 5: mmvs_kernel<T, 5><<<grid, 128, 0, st>>>(w, x, y, K, R, sw, sx, sy); break;
        case 6: mmvs_kernel<T, 6><<<grid, 128, 0, st>>>(w, x, y, K, R, sw, sx, sy); break;
        case 7: mmvs_kernel<T, 7><<<grid, 128, 0, st>>>(w, x, y, K, R, sw, sx, sy); break;
        default: mmvs_kernel<T, 8><<<grid, 128, 0, st>>>(w, x, y, K, R, sw, sx, sy); break;
    }
}

} // namespace

bool ggml_cuda_musa_mmv_small(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_MUSA_MMV_SMALL") == nullptr || atoi(getenv("GGML_MUSA_MMV_SMALL")) != 0;
    if (!enabled) {
        return false;
    }
    if ((src0->type != GGML_TYPE_BF16 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = src0->ne[0], R = src0->ne[1], N = src1->ne[1];
    // measured: faster than mmvf at N=1 (26 vs 31 us, 48x5120 bf16) but slower at N=4 (82 vs 67 us)
    if (R > 512 || N != 1 || src1->ne[0] != K || src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
        src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    const int64_t sw = src0->nb[1]/ggml_type_size(src0->type), sx = src1->nb[1]/sizeof(float), sy = dst->nb[1]/sizeof(float);
    cudaStream_t st = ctx.stream();
    switch (src0->type) {
        case GGML_TYPE_BF16: mmvs_launch(st, (const nv_bfloat16 *) src0->data, (const float *) src1->data, (float *) dst->data, (int) K, (int) R, (int) N, sw, sx, sy); break;
        case GGML_TYPE_F16:  mmvs_launch(st, (const half *) src0->data, (const float *) src1->data, (float *) dst->data, (int) K, (int) R, (int) N, sw, sx, sy); break;
        default:             mmvs_launch(st, (const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) K, (int) R, (int) N, sw, sx, sy); break;
    }
    return true;
}
