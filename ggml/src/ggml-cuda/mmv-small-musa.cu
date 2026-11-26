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

// MTT S80: small f32 x f32 products of activations (Inkling: relative-bias table 1025x24 with K=16, depth mixing
// with K=2304 and a few outputs). These went to muBLAS sgemm: ~100 us each, and muBLAS cannot be captured into a
// MUSA graph. dst[i,j,b2,b3] = sum_k src0[k,i,b2/r2,b3/r3]*src1[k,j,b2,b3]
//   short K: one thread per output;  long K: one 128-thread block per output (strided K + block reduction)
#ifndef MMF_SMALL_SHORT_K
#define MMF_SMALL_SHORT_K 64
#endif
template <typename TX>
static __global__ void mm_f32_small_short(const TX * __restrict__ x, const float * __restrict__ y, float * __restrict__ d,
        const int K, const int M, const int N, const int ne2, const int r2, const int r3,
        const int64_t sx1, const int64_t sx2, const int64_t sx3, const int64_t sy1, const int64_t sy2, const int64_t sy3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x, j = blockIdx.y, b = blockIdx.z;
    if (i >= M) {
        return;
    }
    const int b2 = b % ne2, b3 = b / ne2;
    const TX * xr = x + (b2/r2)*sx2 + (b3/r3)*sx3 + i*sx1;
    const float * yr = y + b2*sy2 + b3*sy3 + j*sy1;
    float s = 0.0f;
    for (int k = 0; k < K; ++k) {
        s = fmaf(ggml_cuda_cast<float>(xr[k]), yr[k], s);
    }
    d[b2*sd2 + b3*sd3 + j*sd1 + i] = s;
}

template <typename TX>
static __global__ void mm_f32_small_long(const TX * __restrict__ x, const float * __restrict__ y, float * __restrict__ d,
        const int K, const int M, const int N, const int ne2, const int r2, const int r3,
        const int64_t sx1, const int64_t sx2, const int64_t sx3, const int64_t sy1, const int64_t sy2, const int64_t sy3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3) {
    __shared__ float red[128];
    const int i = blockIdx.x, j = blockIdx.y, b = blockIdx.z, t = threadIdx.x;
    const int b2 = b % ne2, b3 = b / ne2;
    const TX * xr = x + (b2/r2)*sx2 + (b3/r3)*sx3 + i*sx1;
    const float * yr = y + b2*sy2 + b3*sy3 + j*sy1;
    float s = 0.0f;
    for (int k = t; k < K; k += 128) {
        s = fmaf(ggml_cuda_cast<float>(xr[k]), yr[k], s);
    }
    red[t] = s;
    __syncthreads();
    for (int w = 64; w > 0; w >>= 1) {
        if (t < w) {
            red[t] += red[t + w];
        }
        __syncthreads();
    }
    if (t == 0) {
        d[b2*sd2 + b3*sd3 + j*sd1 + i] = red[0];
    }
    GGML_UNUSED(M); GGML_UNUSED(N);
}

template <typename TX>
static void mm_float_small_launch(cudaStream_t st, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const bool short_k);

bool ggml_cuda_musa_mm_f32_small(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_MUSA_MM_F32_SMALL") == nullptr || atoi(getenv("GGML_MUSA_MM_F32_SMALL")) != 0;
    const bool f32w = src0->type == GGML_TYPE_F32;
    // f16/bf16 weights only while a MUSA graph is being captured (muBLAS cannot be captured)
    bool capturing = false;
    {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        capturing = cudaStreamIsCapturing(ctx.stream(), &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone;
    }
    if (!enabled || !(f32w || (capturing && (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16))) ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    const int64_t ne2 = src1->ne[2], ne3 = src1->ne[3];
    if (src1->ne[0] != K || src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
        dst->ne[0] != M || dst->ne[1] != N || dst->ne[2] != ne2 || dst->ne[3] != ne3 ||
        ne2 % src0->ne[2] != 0 || ne3 % src0->ne[3] != 0 || N > 65535 || ne2*ne3 > 65535) {
        return false;
    }
    const int64_t nout = M*N*ne2*ne3;
    const bool short_k = K <= MMF_SMALL_SHORT_K;
    // long K only for a few outputs (one block each); larger products stay with the regular paths
    // outside a capture only for few outputs (one block each); during a capture anything muBLAS would have taken
    if (!short_k && ((!capturing && nout > 4096) || M > 65535)) {
        return false;
    }
    // K <= 32 (Inkling's relative-bias table: K = 16, 1025 x 24 x 256 outputs per prompt batch) stays here at any
    // size: the generic f32 path took 69 ms for that product
    if (short_k && !capturing && nout > (1 << 22) && (K > 32 || nout > (1LL << 28))) {
        return false;
    }
    switch (src0->type) {
        case GGML_TYPE_F16:  mm_float_small_launch<half>(ctx.stream(), src0, src1, dst, short_k); break;
        case GGML_TYPE_BF16: mm_float_small_launch<nv_bfloat16>(ctx.stream(), src0, src1, dst, short_k); break;
        default:             mm_float_small_launch<float>(ctx.stream(), src0, src1, dst, short_k); break;
    }
    return true;
}

template <typename TX>
static void mm_float_small_launch(cudaStream_t st, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const bool short_k) {
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1], ne2 = src1->ne[2], ne3 = src1->ne[3];
    const int r2 = (int) (ne2/src0->ne[2]), r3 = (int) (ne3/src0->ne[3]);
    const int64_t fx = ggml_type_size(src0->type), f = sizeof(float);
    if (short_k) {
        const dim3 grid((unsigned) ((M + 127)/128), (unsigned) N, (unsigned) (ne2*ne3));
        mm_f32_small_short<TX><<<grid, 128, 0, st>>>((const TX *) src0->data, (const float *) src1->data, (float *) dst->data,
            (int) K, (int) M, (int) N, (int) ne2, r2, r3, src0->nb[1]/fx, src0->nb[2]/fx, src0->nb[3]/fx,
            src1->nb[1]/f, src1->nb[2]/f, src1->nb[3]/f, dst->nb[1]/f, dst->nb[2]/f, dst->nb[3]/f);
    } else {
        const dim3 grid((unsigned) M, (unsigned) N, (unsigned) (ne2*ne3));
        mm_f32_small_long<TX><<<grid, 128, 0, st>>>((const TX *) src0->data, (const float *) src1->data, (float *) dst->data,
            (int) K, (int) M, (int) N, (int) ne2, r2, r3, src0->nb[1]/fx, src0->nb[2]/fx, src0->nb[3]/fx,
            src1->nb[1]/f, src1->nb[2]/f, src1->nb[3]/f, dst->nb[1]/f, dst->nb[2]/f, dst->nb[3]/f);
    }
}
