#include "argsort.cuh"
#include "top-k.cuh"

// Adjusted implementation thresholds from #28547, can be overridden at build time
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
// not measured on HIP/MUSA, keep the old split
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 1024
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 512
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC

#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT
#    define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT 4096
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT

// bitonic up to this width while nrows fits in one wave of SMs, 0 disables
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 0
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 1024
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

// max rows for the per-row DeviceTopK / CUB argsort path before switching to radix / bitonic
#ifndef GGML_CUDA_TOP_K_NROWS_THRESHOLD
#    ifdef CUB_TOP_K_AVAILABLE
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 2
#    else
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 1
#    endif
#endif // GGML_CUDA_TOP_K_NROWS_THRESHOLD

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#endif                            // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

// one state per cache line, so the gather atomics of different rows do not compete
struct alignas(128) top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int block_cols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    const int64_t col_begin = (int64_t) row_block * block_cols;
    const int64_t col_end   = min(col_begin + block_cols, (int64_t) ncols);
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int64_t col = col_begin + tid; col < col_end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int block_cols,
        int k,
        int blocks_per_row) {
    constexpr int NBINS     = 1 << RADIX_BITS;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = BLOCK_SIZE / warp_size;
    __shared__ int warp_ties[nwarps];

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];
    const uint32_t threshold = state->prefix;
    const int rank = state->rank;

    // each warp reads a contiguous part of the contiguous range of the block
    const int     warp_cols  = block_cols / nwarps;
    const int64_t warp_begin = (int64_t) row_block * block_cols + warp * warp_cols;
    const int64_t warp_end   = min(warp_begin + warp_cols, (int64_t) ncols);

    int nties = 0;
    for (int64_t col = warp_begin + lane; col < warp_end; col += warp_size) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > threshold) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == threshold) {
            nties++;
        }
    }

    // the ties fill the last rank slots in column order, the lowest columns win.
    // the histograms of the last pass count the ties of each block in the bin of the threshold.
    const int * block_ties = block_histograms + (size_t) row * blocks_per_row * NBINS + (threshold & (NBINS - 1));
    int pos = 0;
    for (int b = lane; b < row_block; b += warp_size) {
        pos += block_ties[b * NBINS];
    }
    pos = warp_reduce_sum<warp_size>(pos);
    if (pos >= rank || block_ties[row_block * NBINS] == 0) {
        return; // the same result in all threads of the block
    }

    nties = warp_reduce_sum<warp_size>(nties);
    if (lane == 0) {
        warp_ties[warp] = nties;
    }
    __syncthreads();
    for (int w = 0; w < warp; ++w) {
        pos += warp_ties[w];
    }

    // pos is the same in all lanes of the warp, so the lanes leave the loop together
    const int pos_end = min(pos + nties, rank);
    for (int64_t tile = warp_begin; tile < warp_end && pos < pos_end; tile += warp_size) {
        const int64_t col    = tile + lane;
        const bool    is_tie = col < warp_end && top_k_float_to_ordered(row_src[col]) == threshold;

        const ggml_cuda_thread_mask_t ties = ggml_cuda_ballot(is_tie);
        const int tie_pos = pos + ggml_cuda_popc(ties & (((ggml_cuda_thread_mask_t) 1 << lane) - 1));
        if (is_tie && tie_pos < pos_end) {
            row_dst[k - rank + tie_pos] = col;
        }
        pos += ggml_cuda_popc(ties);
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);
    // each block of a row reads a contiguous range of whole tiles
    const int block_cols = (int) (((int64_t) ncols + blocks_per_row * BLOCK_SIZE - 1) / (blocks_per_row * BLOCK_SIZE) * BLOCK_SIZE);

    // chunk the rows to bound the histogram memory to 64 MB
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, chunk_nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) chunk_nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        top_k_radix_init<<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows, k);

        const dim3 row_grid(blocks_per_row * iter_nrows);
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, ncols, block_cols, blocks_per_row, shift);
            top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
                <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
        }

        top_k_radix_gather<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, histograms, states, ncols, block_cols, k, blocks_per_row);

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

static void top_k_argsort_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, bool use_cub, cudaStream_t stream) {
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) ncols * sizeof(int), nrows);

    ggml_cuda_pool_alloc<int> tmp_alloc(pool, (size_t) ncols * chunk_nrows);
    int * tmp = tmp_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        if (use_cub) {
#ifdef GGML_CUDA_USE_CUB
            argsort_f32_i32_cuda_cub(pool, src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
#else
            GGML_ABORT("CUB is not available");
#endif // GGML_CUDA_USE_CUB
        } else {
            argsort_f32_i32_cuda_bitonic(src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

#ifdef GGML_USE_MUSA
// MUSA top-k for k <= 64 (GGML_MUSA_TOPK=0 disables): the radix path above serializes on shared atomics
// (logits share their high bits) and costs ~0.5 ms per 200k row. Here:
//   1) max of every CH-column chunk
//   2) one block per row: sort the chunk maxima, T = k-th largest; every top-k element is >= T and lies in a
//      chunk whose max is >= T; gather those elements (normally ~k..2k) and sort them.
// overflow (many ties / -inf rows) falls back to an exact k-round selection in the same block.
#define MTK_CAP  2048
#define MTK_NT   512
#define MTK_MAXC 256

static __device__ __forceinline__ void mtk_bitonic_desc(float * key, int * idx, int n) {
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int t = threadIdx.x; t < n/2; t += MTK_NT) {
                const int i = 2*t - (t & (stride - 1));
                const int j = i + stride;
                const bool desc = (i & size) == 0;
                const float a = key[i], b = key[j];
                if (desc ? (a < b) : (a > b)) {
                    key[i] = b; key[j] = a;
                    const int ti = idx[i]; idx[i] = idx[j]; idx[j] = ti;
                }
            }
            __syncthreads();
        }
    }
}

template <int WS>
static __global__ void mtk_chunk_max(const float * __restrict__ src, float * __restrict__ cmax, int ncols, int nchunks, int ch) {
    const int row  = blockIdx.y;
    const int warp = blockIdx.x * (MTK_NT / WS) + threadIdx.x / WS;
    const int lane = threadIdx.x % WS;
    if (warp >= nchunks) {
        return;
    }
    const float * r = src + (size_t) row * ncols;
    const int c0 = warp * ch;
    const int c1 = min(c0 + ch, ncols);
    float m = -INFINITY;
    for (int c = c0 + lane; c < c1; c += WS) {
        m = fmaxf(m, r[c]);
    }
    m = warp_reduce_max<WS>(m);
    if (lane == 0) {
        cmax[(size_t) row * nchunks + warp] = m;
    }
}

static __global__ void __launch_bounds__(MTK_NT) mtk_select(const float * __restrict__ src, const float * __restrict__ cmax,
        int * __restrict__ dst, int ncols, int nchunks, int ch, int k) {
    __shared__ float skey[MTK_CAP];
    __shared__ int   sidx[MTK_CAP];
    __shared__ int   clist[MTK_MAXC];
    __shared__ int   cnt;
    __shared__ int   nlist;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * r = src + (size_t) row * ncols;
    int * out = dst + (size_t) row * k;

    bool overflow = false;
    if (nchunks > 0) {
        int n2 = 1; while (n2 < nchunks) n2 <<= 1;
        for (int i = tid; i < n2; i += MTK_NT) {
            skey[i] = i < nchunks ? cmax[(size_t) row * nchunks + i] : -INFINITY;
            sidx[i] = i;
        }
        if (tid == 0) { cnt = 0; nlist = 0; }
        __syncthreads();
        mtk_bitonic_desc(skey, sidx, n2);
        const float T = skey[k - 1];
        // chunks with max >= T lead the sorted maxima
        for (int i = tid; i < nchunks && i < MTK_MAXC + 1; i += MTK_NT) {
            if (skey[i] >= T) {
                atomicAdd(&nlist, 1);
                if (i < MTK_MAXC) clist[i] = sidx[i];
            }
        }
        __syncthreads();
        const int m = nlist;
        overflow = m > MTK_MAXC || T == -INFINITY;
        __syncthreads();
        if (!overflow) {
            for (int e = tid; e < m*ch; e += MTK_NT) {
                const int col = clist[e / ch] * ch + e % ch;
                if (col < ncols) {
                    const float v = r[col];
                    if (v >= T) {
                        const int p = atomicAdd(&cnt, 1);
                        if (p < MTK_CAP) { skey[p] = v; sidx[p] = col; }
                    }
                }
            }
            __syncthreads();
            overflow = cnt > MTK_CAP;
        }
    } else {
        for (int i = tid; i < ncols; i += MTK_NT) { skey[i] = r[i]; sidx[i] = i; }
        if (tid == 0) cnt = ncols;
        __syncthreads();
    }

    if (!overflow) {
        const int c = cnt;
        int n2 = 1; while (n2 < c) n2 <<= 1;
        for (int i = c + tid; i < n2; i += MTK_NT) { skey[i] = -INFINITY; sidx[i] = -1; }
        __syncthreads();
        mtk_bitonic_desc(skey, sidx, n2);
        for (int i = tid; i < k; i += MTK_NT) out[i] = sidx[i];
        return;
    }

    // exact fallback: k rounds, each the largest (value desc, column asc) below the previous pick
    __shared__ float wv[MTK_NT];
    __shared__ int   wi[MTK_NT];
    float pv = INFINITY; int pi = -1;
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY; int bi = INT_MAX;
        for (int c = tid; c < ncols; c += MTK_NT) {
            const float v = r[c];
            const bool below = v < pv || (v == pv && c > pi);
            if (below && (v > bv || (v == bv && c < bi))) { bv = v; bi = c; }
        }
        wv[tid] = bv; wi[tid] = bi;
        __syncthreads();
        for (int st = MTK_NT/2; st > 0; st >>= 1) {
            if (tid < st) {
                const float ov = wv[tid + st]; const int oi = wi[tid + st];
                if (ov > wv[tid] || (ov == wv[tid] && oi < wi[tid])) { wv[tid] = ov; wi[tid] = oi; }
            }
            __syncthreads();
        }
        pv = wv[0]; pi = wi[0];
        if (tid == 0) out[j] = pi == INT_MAX ? 0 : pi;
        __syncthreads();
    }
}

static bool top_k_musa_small(ggml_cuda_pool & pool, const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    static const bool on = getenv("GGML_MUSA_TOPK") == nullptr || atoi(getenv("GGML_MUSA_TOPK")) != 0;
    if (!on || k > 64 || k < 1 || nrows > 65535 || ncols < k) {
        return false;
    }
    if (ncols <= MTK_CAP) {
        mtk_select<<<(int) nrows, MTK_NT, 0, stream>>>(src, nullptr, dst, ncols, 0, 0, k);
        return true;
    }
    int ch = 128;
    while ((ncols + ch - 1) / ch > 512) ch *= 2;
    const int nchunks = (ncols + ch - 1) / ch;
    if (nchunks < k) {
        return false;
    }
    constexpr int WS = 32; // ggml warp helpers use 32-lane logical warps on MUSA
    ggml_cuda_pool_alloc<float> cmax(pool, (size_t) nchunks * nrows);
    const dim3 g1((nchunks + MTK_NT/WS - 1) / (MTK_NT/WS), (unsigned) nrows);
    mtk_chunk_max<WS><<<g1, MTK_NT, 0, stream>>>(src, cmax.get(), ncols, nchunks, ch);
    mtk_select<<<(int) nrows, MTK_NT, 0, stream>>>(src, cmax.get(), dst, ncols, nchunks, ch, k);
    return true;
}
#endif // GGML_USE_MUSA

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    const int device = ggml_cuda_get_device();

#ifdef GGML_USE_MUSA
    if (top_k_musa_small(pool, src0_d, dst_d, (int) ncols, nrows, (int) k, stream)) {
        return;
    }
#endif // GGML_USE_MUSA

#ifdef CUB_TOP_K_AVAILABLE
    // a single row always uses DeviceTopK if available
    const bool bitonic_short    = nrows > 1 && ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#else
    const bool bitonic_short    = ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#endif // CUB_TOP_K_AVAILABLE
    const bool bitonic_few_rows = nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD &&
                                  ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS &&
                                  nrows <= ggml_cuda_info().devices[device].nsm;

    if (bitonic_short || bitonic_few_rows) {
        // the padded row must fit in shared memory
        const int ncols_pad = next_power_of_2(ncols);
        if (ncols_pad * sizeof(int) <= ggml_cuda_info().devices[device].smpb) {
            top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, false, stream);
            return;
        }
    }

    if (nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Assess perf of `DeviceBatchedTopK` for multi-row TopK & CCCL >= 3.5.0, re-running perf sweep of https://github.com/ggml-org/llama.cpp/pull/28713
    for (int64_t i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    if (ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT) {
        top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, true, stream);
    } else {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    }
#else                             // GGML_CUDA_USE_CUB
    top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
#endif                            // CUB_TOP_K_AVAILABLE
}
