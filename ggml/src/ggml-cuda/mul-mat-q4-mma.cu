// MTT S80 (mp_21): quantized matrix x (1..N tokens) on the int8 tensor cores.
// Types: Q4_0, Q4_1, Q5_K, Q8_0, IQ4_XS, IQ3_S, IQ3_XXS, Q4_K, IQ2_S, Q2_K, Q6_K, IQ2_XXS, IQ2_XS.
//
// Native MMA on mp_21: __musa_imma_m16n8k16_mma, one 16x8x16 tile per 128-lane warp, UNSIGNED int8 only.
//   A (16 rows x 16 k): lanes 0-63, a[0]; row = lane/4, k = 4*(lane%4) + byte
//   B (16 k x 8 cols) : lanes 0-31, b[0]; col = lane/4, k = 4*(lane%4) + byte
//   D (16 x 8 int32)  : d[0];            row = lane/8, col = lane%8
// Signed activations are offset by +128 (y ^ 0x80); column 7 of B is all ones so the MMA also
// returns sum(A) per row, which removes that offset: sum(A*y) = D - 128*sum(A).
// Weights are made unsigned per type: Q4_0 nibbles as-is (zero point 8), IQ4_XS kvalues+128,
// IQ3_S / IQ3_XXS 128+sign*grid. Each type's constant offset is removed with sum(y).
//
// Weights are rewritten in place on first use into a tiled layout of the same size: tile t (16 rows)
// occupies exactly the bytes that rows 16t..16t+15 occupied.
//   Q4_0  : per superblock (8 blocks, 256 k): QS[sb][64 lanes] 2 x uint4 (word j = qs[4c..] of block j) | D[sb][16 rows] 8 halves
//   Q4_1  : as Q4_0 with DM[sb][16 rows] 8 half2 (d, m)
//   Q5_K  : QS as Q4_K | QH[sb][64 lanes] uint2 (qh[4c..], qh[16+4c..]) | SC[sb][16 rows] 16 B (d, dmin, scales[12])
//   IQ4_XS: per superblock: [64 lanes x 2 uint4 nibbles] ... then [16 rows x u64 (d | 6-bit scales<<16)]
//   IQ3_S : per superblock: QSH[h][64] uint2 | SGB[64] uint2 (bit-sliced: bit 8e+(k&7) of word k>>3 = sign e of item k=2ib+h) | QH[64] u16 ... then D[16] half | SC[16] u32
//   IQ3_XXS: QSH[sb][h][64] uint2 (index bytes) | AUX[sb][16 rows][8] u32 (signs+scale words) | D[sb][16] half
//   Q4_K  : QS[sb][64 lanes][8] u32 (word 2p+h = qs[32p+16h+4c..]) | SC[sb][16 rows] 16 B (d, dmin, scales[12])
//   IQ2_S : IDX[sb][16 rows][2 lane-pairs] uint4 | SGN[same] uint4 | QH[sb][16][2] u32 (2 bits per (ib,h)) | SC[sb][16] 8 B | D[sb][16] half
// Activations (own quantizer, q8_1 math, fp32 scales), per token:
//   ya[g][c][8 x u32]: for j=0..3: y[4g+j][4c..4c+3]^0x80, y[4g+j][16+4c..16+4c+3]^0x80
//   da[g][4 x float2]: (d, sum(x)) per 32-block
//
// Default on for MUSA; GGML_MUSA_Q4_MMA=0 disables everything, GGML_MUSA_QMMA_TYPES=q4_0,iq4_xs,... selects types.
#include "common.cuh"
#include "ggml-cuda.h"
#include <unordered_set>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <string>

#define QMMA_MAX_TOK 7       // tokens per MMA pass (column 7 is the ones column)

namespace {

std::unordered_set<const void *> g_tiled;
std::mutex g_tiled_lock;

bool qmma_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_MUSA_Q4_MMA");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

bool qmma_type_enabled(ggml_type t) {
    static const std::string types = [] {
        const char * e = getenv("GGML_MUSA_QMMA_TYPES");
        return std::string(e ? e : "iq4_xs,iq3_s,iq3_xxs,q4_k,iq2_s,q2_k,q6_k,iq2_xxs,iq2_xs,q8_0,q4_0,q4_1,q5_k");
    }();
    switch (t) {
        case GGML_TYPE_Q4_0:   return types.find("q4_0")   != std::string::npos;
        case GGML_TYPE_Q4_1:   return types.find("q4_1")   != std::string::npos;
        case GGML_TYPE_Q5_K:   return types.find("q5_k")   != std::string::npos;
        case GGML_TYPE_IQ4_XS: return types.find("iq4_xs") != std::string::npos;
        case GGML_TYPE_IQ3_S:  return types.find("iq3_s")  != std::string::npos;
        case GGML_TYPE_IQ3_XXS:return types.find("iq3_xxs")!= std::string::npos;
        case GGML_TYPE_Q4_K:   return types.find("q4_k")   != std::string::npos;
        case GGML_TYPE_IQ2_S:  return types.find("iq2_s")  != std::string::npos;
        case GGML_TYPE_Q2_K:   return types.find("q2_k")   != std::string::npos;
        case GGML_TYPE_Q6_K:   return types.find("q6_k")   != std::string::npos;
        case GGML_TYPE_IQ2_XXS:return types.find("iq2_xxs")!= std::string::npos;
        case GGML_TYPE_IQ2_XS: return types.find("iq2_xs") != std::string::npos;
        case GGML_TYPE_Q8_0:   return types.find("q8_0")   != std::string::npos;
        default:               return false;
    }
}

// =============================================================================================
// repack kernels (src = scratch copy of a chunk of tiles, dst = tensor bytes of the same tiles)

// Q4_0 / Q4_1 (8 blocks per 256-k superblock): one thread per (tile, superblock, lane) for the nibble words,
// per (tile, superblock, row) for the scales. BLK = block_q4_0 / block_q4_1; scales: Q4_0 8 halves (16 B), Q4_1 8 half2 (32 B)
template <class BLK>
__global__ void qmma_repack_q4_x_q(const BLK * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*64) {
        return;
    }
    const int L = (int) (i % 64), sb = (int) ((i / 64) % nsb), t = (int) (i / (64*nsb));
    const int row = L >> 2, c = L & 3, nb = nsb*8;
    const BLK * rb = src + ((int64_t) t*16 + row)*nb + sb*8;
    uint32_t w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint8_t * q = rb[j].qs + 4*c;
        w[j] = q[0] | (q[1] << 8) | (q[2] << 16) | ((uint32_t) q[3] << 24);
    }
    uint4 * o = (uint4 *) (dst + (int64_t) t*16*nb*sizeof(BLK));
    o[((int64_t) sb*64 + L)*2]     = make_uint4(w[0], w[1], w[2], w[3]);
    o[((int64_t) sb*64 + L)*2 + 1] = make_uint4(w[4], w[5], w[6], w[7]);
}
__global__ void qmma_repack_q4_0_d(const block_q4_0 * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*16) {
        return;
    }
    const int row = (int) (i % 16), sb = (int) ((i / 16) % nsb), t = (int) (i / (16*nsb));
    const int nb = nsb*8;
    const block_q4_0 * rb = src + ((int64_t) t*16 + row)*nb + sb*8;
    uint16_t h[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        h[j] = __half_as_ushort(rb[j].d);
    }
    uint4 * o = (uint4 *) (dst + (int64_t) t*16*nb*sizeof(block_q4_0) + (int64_t) nsb*2048);
    o[sb*16 + row] = make_uint4(h[0] | ((uint32_t) h[1] << 16), h[2] | ((uint32_t) h[3] << 16),
                                h[4] | ((uint32_t) h[5] << 16), h[6] | ((uint32_t) h[7] << 16));
}
__global__ void qmma_repack_q4_1_d(const block_q4_1 * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*16) {
        return;
    }
    const int row = (int) (i % 16), sb = (int) ((i / 16) % nsb), t = (int) (i / (16*nsb));
    const int nb = nsb*8;
    const block_q4_1 * rb = src + ((int64_t) t*16 + row)*nb + sb*8;
    uint32_t h[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        h[j] = *(const uint32_t *) &rb[j];      // d | m << 16
    }
    uint4 * o = (uint4 *) (dst + (int64_t) t*16*nb*sizeof(block_q4_1) + (int64_t) nsb*2048);
    o[(sb*16 + row)*2]     = make_uint4(h[0], h[1], h[2], h[3]);
    o[(sb*16 + row)*2 + 1] = make_uint4(h[4], h[5], h[6], h[7]);
}

// IQ4_XS: one thread per (tile, superblock, lane) and per (tile, superblock, row)
__global__ void qmma_repack_iq4_xs_q(const block_iq4_xs * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*64) {
        return;
    }
    const int lane = i % 64, sb = (i / 64) % nsb, t = i / (64*nsb);
    const int row = t*16 + lane/4, c = lane % 4;
    const block_iq4_xs * b = src + (int64_t) row*nsb + sb;
    uint32_t w[8];
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const uint8_t * q = b->qs + 16*ib + 4*c;
        w[ib] = q[0] | (q[1] << 8) | (q[2] << 16) | ((uint32_t) q[3] << 24);
    }
    uint4 * o = (uint4 *) (dst + (int64_t) t*16*nsb*sizeof(block_iq4_xs) + ((int64_t) sb*64 + lane)*32);
    o[0] = make_uint4(w[0], w[1], w[2], w[3]);
    o[1] = make_uint4(w[4], w[5], w[6], w[7]);
}
__global__ void qmma_repack_iq4_xs_d(const block_iq4_xs * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*16) {
        return;
    }
    const int r = i % 16, sb = (i / 16) % nsb, t = i / (16*nsb);
    const block_iq4_xs * b = src + (int64_t) (t*16 + r)*nsb + sb;
    uint64_t v = __half_as_ushort(b->d);
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const uint64_t ls = ((b->scales_l[ib/2] >> (4*(ib%2))) & 0xF) | (((b->scales_h >> (2*ib)) & 3) << 4);
        v |= ls << (16 + 6*ib);
    }
    uint32_t * o = (uint32_t *) (dst + (int64_t) t*16*nsb*sizeof(block_iq4_xs) + (int64_t) nsb*2048 + ((int64_t) sb*16 + r)*8);
    o[0] = (uint32_t) v;
    o[1] = (uint32_t) (v >> 32);
}

// IQ3_S
__device__ __forceinline__ size_t iq3_o_sg(int nsb) { return (size_t) nsb*1024; }
__device__ __forceinline__ size_t iq3_o_qh(int nsb) { return (size_t) nsb*1536; }
__device__ __forceinline__ size_t iq3_o_d (int nsb) { return (size_t) nsb*1664; }
__device__ __forceinline__ size_t iq3_o_sc(int nsb) { return (size_t) nsb*1696; }

__global__ void qmma_repack_iq3_s_q(const block_iq3_s * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*64) {
        return;
    }
    const int lane = i % 64, sb = (i / 64) % nsb, t = i / (64*nsb);
    const int row = t*16 + lane/4, c = lane % 4;
    const block_iq3_s * b = src + (int64_t) row*nsb + sb;
    uint8_t * tb = dst + (int64_t) t*16*nsb*sizeof(block_iq3_s);
    uint32_t qhb = 0;
    uint32_t sgb[2] = {0, 0};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint32_t qlo = 0, qhi = 0;
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            const int li = 4*h + c, k = 2*ib + h;
            const uint32_t qb = b->qs[8*ib + li];
            if (ib < 4) { qlo |= qb << (8*ib); } else { qhi |= qb << (8*(ib - 4)); }
            qhb |= ((b->qh[ib] >> li) & 1u) << k;
            const uint32_t nib = (b->signs[4*ib + li/2] >> (4*(li%2))) & 0xF;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                sgb[k >> 3] |= ((nib >> e) & 1u) << (8*e + (k & 7));
            }
        }
        *(uint2 *) (tb + ((int64_t) (sb*2 + h)*64 + lane)*8) = make_uint2(qlo, qhi);
    }
    *(uint2 *) (tb + iq3_o_sg(nsb) + ((int64_t) sb*64 + lane)*8) = make_uint2(sgb[0], sgb[1]);
    *(uint16_t *) (tb + iq3_o_qh(nsb) + ((int64_t) sb*64 + lane)*2) = (uint16_t) qhb;
}
__global__ void qmma_repack_iq3_s_d(const block_iq3_s * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*16) {
        return;
    }
    const int r = i % 16, sb = (i / 16) % nsb, t = i / (16*nsb);
    const block_iq3_s * b = src + (int64_t) (t*16 + r)*nsb + sb;
    uint8_t * tb = dst + (int64_t) t*16*nsb*sizeof(block_iq3_s);
    *(half *) (tb + iq3_o_d(nsb) + ((int64_t) sb*16 + r)*2) = b->d;
    *(uint32_t *) (tb + iq3_o_sc(nsb) + ((int64_t) sb*16 + r)*4) =
        b->scales[0] | (b->scales[1] << 8) | (b->scales[2] << 16) | ((uint32_t) b->scales[3] << 24);
}

// IQ3_XXS
__global__ void qmma_repack_iq3_xxs_q(const block_iq3_xxs * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*64) {
        return;
    }
    const int lane = i % 64, sb = (i / 64) % nsb, t = i / (64*nsb);
    const int row = t*16 + lane/4, c = lane % 4;
    const block_iq3_xxs * b = src + (int64_t) row*nsb + sb;
    uint8_t * tb = dst + (int64_t) t*16*nsb*sizeof(block_iq3_xxs);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            const uint32_t qb = b->qs[8*ib + 4*h + c];
            if (ib < 4) { lo |= qb << (8*ib); } else { hi |= qb << (8*(ib - 4)); }
        }
        *(uint2 *) (tb + ((int64_t) (sb*2 + h)*64 + lane)*8) = make_uint2(lo, hi);
    }
}
__global__ void qmma_repack_iq3_xxs_d(const block_iq3_xxs * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*16) {
        return;
    }
    const int r = i % 16, sb = (i / 16) % nsb, t = i / (16*nsb);
    const block_iq3_xxs * b = src + (int64_t) (t*16 + r)*nsb + sb;
    uint8_t * tb = dst + (int64_t) t*16*nsb*sizeof(block_iq3_xxs);
    uint32_t * aux = (uint32_t *) (tb + (int64_t) nsb*1024 + ((int64_t) sb*16 + r)*32);
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const uint8_t * q = b->qs + 64 + 4*ib;
        aux[ib] = q[0] | (q[1] << 8) | (q[2] << 16) | ((uint32_t) q[3] << 24);
    }
    *(half *) (tb + (int64_t) nsb*1536 + ((int64_t) sb*16 + r)*2) = b->d;
}

// Q4_K
__global__ void qmma_repack_q4_K_q(const block_q4_K * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*64) {
        return;
    }
    const int lane = i % 64, sb = (i / 64) % nsb, t = i / (64*nsb);
    const int row = t*16 + lane/4, c = lane % 4;
    const block_q4_K * b = src + (int64_t) row*nsb + sb;
    uint32_t w[8];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint8_t * q = b->qs + 32*p + 16*h + 4*c;
            w[2*p + h] = q[0] | (q[1] << 8) | (q[2] << 16) | ((uint32_t) q[3] << 24);
        }
    }
    uint4 * o = (uint4 *) (dst + (int64_t) t*16*nsb*sizeof(block_q4_K) + ((int64_t) sb*64 + lane)*32);
    o[0] = make_uint4(w[0], w[1], w[2], w[3]);
    o[1] = make_uint4(w[4], w[5], w[6], w[7]);
}
__global__ void qmma_repack_q4_K_d(const block_q4_K * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*16) {
        return;
    }
    const int r = i % 16, sb = (i / 16) % nsb, t = i / (16*nsb);
    const uint32_t * b = (const uint32_t *) (src + (int64_t) (t*16 + r)*nsb + sb);    // first 16 bytes: dm + scales[12]
    uint4 * o = (uint4 *) (dst + (int64_t) t*16*nsb*sizeof(block_q4_K) + (int64_t) nsb*2048 + ((int64_t) sb*16 + r)*16);
    *o = make_uint4(b[0], b[1], b[2], b[3]);
}

// Q5_K: QS like Q4_K (nsb*2048) | QH[sb][64 lanes] uint2 (nsb*512) | SC[sb][16] 16 B (nsb*256) = 16*176 per superblock
__global__ void qmma_repack_q5_K_q(const block_q5_K * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*64) {
        return;
    }
    const int lane = i % 64, sb = (i / 64) % nsb, t = i / (64*nsb);
    const int row = t*16 + lane/4, c = lane % 4;
    const block_q5_K * b = src + (int64_t) row*nsb + sb;
    uint32_t w[8];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint8_t * q = b->qs + 32*p + 16*h + 4*c;
            w[2*p + h] = q[0] | (q[1] << 8) | (q[2] << 16) | ((uint32_t) q[3] << 24);
        }
    }
    uint8_t * tb = dst + (int64_t) t*16*nsb*sizeof(block_q5_K);
    uint4 * o = (uint4 *) (tb + ((int64_t) sb*64 + lane)*32);
    o[0] = make_uint4(w[0], w[1], w[2], w[3]);
    o[1] = make_uint4(w[4], w[5], w[6], w[7]);
    const uint8_t * qh0 = b->qh + 4*c, * qh1 = b->qh + 16 + 4*c;
    *(uint2 *) (tb + (int64_t) nsb*2048 + ((int64_t) sb*64 + lane)*8) =
        make_uint2(qh0[0] | (qh0[1] << 8) | (qh0[2] << 16) | ((uint32_t) qh0[3] << 24),
                   qh1[0] | (qh1[1] << 8) | (qh1[2] << 16) | ((uint32_t) qh1[3] << 24));
}
__global__ void qmma_repack_q5_K_d(const block_q5_K * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*16) {
        return;
    }
    const int r = i % 16, sb = (i / 16) % nsb, t = i / (16*nsb);
    const uint32_t * b = (const uint32_t *) (src + (int64_t) (t*16 + r)*nsb + sb);    // first 16 bytes: dm + scales[12]
    uint4 * o = (uint4 *) (dst + (int64_t) t*16*nsb*sizeof(block_q5_K) + (int64_t) nsb*2560 + ((int64_t) sb*16 + r)*16);
    *o = make_uint4(b[0], b[1], b[2], b[3]);
}

// IQ2_S: one thread per (tile, superblock, row)
__global__ void qmma_repack_iq2_s(const block_iq2_s * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*16) {
        return;
    }
    const int r = i % 16, sb = (i / 16) % nsb, t = i / (16*nsb);
    const block_iq2_s * b = src + (int64_t) (t*16 + r)*nsb + sb;
    uint8_t * tb = dst + (int64_t) t*16*nsb*sizeof(block_iq2_s);
#pragma unroll
    for (int cp = 0; cp < 2; ++cp) {
        uint32_t iw[4] = {0, 0, 0, 0}, sw[4] = {0, 0, 0, 0}, q32 = 0;
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int bi = 2*ib + h;
                iw[bi >> 2] |= (uint32_t) b->qs[4*ib + 2*h + cp]      << (8*(bi & 3));
                sw[bi >> 2] |= (uint32_t) b->qs[32 + 4*ib + 2*h + cp] << (8*(bi & 3));
                q32 |= (uint32_t) ((b->qh[ib] >> (2*(2*h + cp))) & 3) << (2*bi);
            }
        }
        *(uint4 *) (tb + (((int64_t) sb*16 + r)*2 + cp)*16)                     = make_uint4(iw[0], iw[1], iw[2], iw[3]);
        *(uint4 *) (tb + (int64_t) nsb*512 + (((int64_t) sb*16 + r)*2 + cp)*16) = make_uint4(sw[0], sw[1], sw[2], sw[3]);
        *(uint32_t *) (tb + (int64_t) nsb*1024 + (((int64_t) sb*16 + r)*2 + cp)*4) = q32;
    }
    uint32_t s0 = 0, s1 = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        s0 |= (uint32_t) b->scales[k] << (8*k);
        s1 |= (uint32_t) b->scales[4 + k] << (8*k);
    }
    *(uint2 *) (tb + (int64_t) nsb*1152 + ((int64_t) sb*16 + r)*8) = make_uint2(s0, s1);
    *(half *) (tb + (int64_t) nsb*1280 + ((int64_t) sb*16 + r)*2) = b->d;
}

// Q2_K : QW[sb][64 lanes] uint4 (word n*2+h = qs[32n+16h+4c..]) | SC[sb][16] 16 B | DM[sb][16] 4 B          (1344 = 16*84)
// Q6_K : QL[sb][64][8] u32 (n*4+part*2+h = ql[64n+32part+16h+4c..]) | QH[sb][64][4] u32 (n*2+h = qh[32n+16h+4c..])
//        | SC[sb][16] 16 B int8 | D[sb][16] half                                                               (3360 = 16*210)
// IQ2_XXS: IDX[sb][16 rows][8] u32 (index bytes of sub-block ib) | AUX[sb][16][8] u32 (signs+scale) | D[sb][16] (1056 = 16*66)
// IQ2_XS : QS[sb][16 rows][2 lane-pairs][16] u16 (item 2ib+h = qs[4ib+2h+cp]) | SC[sb][16] 8 B | D[sb][16]     (1184 = 16*74)
__device__ __forceinline__ uint32_t qmma_ld32(const uint8_t * p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t) p[3] << 24);
}

// Q8_0 tile layout (per 256-k superblock sb): QS[sb][lane 0..63][gg 0..1][2] uint4 -- lane (row L/4, c L%4) holds,
// for the 4 blocks j of half gg, the words a0 = q[32j + 4c ..], a1 = q[32j + 16 + 4c ..]; then D[sb][16 rows] = 8 halves.
// nsb*4096 + nsb*256 bytes = 16 rows * K/32 * 34: in place.
__global__ void qmma_repack_q8_0_q(const block_q8_0 * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*64) {
        return;
    }
    const int L = (int) (i % 64), sb = (int) ((i / 64) % nsb), t = (int) (i / (64*nsb));
    const int row = L >> 2, c = L & 3, nb = nsb*8;
    const block_q8_0 * rb = src + ((int64_t) t*16 + row)*nb;
    uint4 * o = (uint4 *) (dst + (int64_t) t*16*nb*sizeof(block_q8_0));
#pragma unroll
    for (int gg = 0; gg < 2; ++gg) {
        uint32_t w[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t * qs = (const uint8_t *) rb[sb*8 + gg*4 + j].qs;
            w[2*j]     = qmma_ld32(qs + 4*c);
            w[2*j + 1] = qmma_ld32(qs + 16 + 4*c);
        }
        o[(((int64_t) sb*64 + L)*2 + gg)*2]     = make_uint4(w[0], w[1], w[2], w[3]);
        o[(((int64_t) sb*64 + L)*2 + gg)*2 + 1] = make_uint4(w[4], w[5], w[6], w[7]);
    }
}
__global__ void qmma_repack_q8_0_d(const block_q8_0 * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*16) {
        return;
    }
    const int row = (int) (i % 16), sb = (int) ((i / 16) % nsb), t = (int) (i / (16*nsb));
    const int nb = nsb*8;
    const block_q8_0 * rb = src + ((int64_t) t*16 + row)*nb + sb*8;
    uint16_t h[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        h[j] = __half_as_ushort(rb[j].d);
    }
    uint4 * o = (uint4 *) (dst + (int64_t) t*16*nb*sizeof(block_q8_0) + (int64_t) nsb*4096);
    o[sb*16 + row] = make_uint4(h[0] | ((uint32_t) h[1] << 16), h[2] | ((uint32_t) h[3] << 16),
                                h[4] | ((uint32_t) h[5] << 16), h[6] | ((uint32_t) h[7] << 16));
}
__global__ void qmma_repack_q2_K(const block_q2_K * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*64) {
        return;
    }
    const int lane = i % 64, sb = (i / 64) % nsb, t = i / (64*nsb);
    const int c = lane % 4;
    uint8_t * tb = dst + (int64_t) t*16*nsb*sizeof(block_q2_K);
    {
        const block_q2_K * b = src + (int64_t) (t*16 + lane/4)*nsb + sb;
        uint32_t w[4];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                w[2*n + h] = qmma_ld32(b->qs + 32*n + 16*h + 4*c);
            }
        }
        *(uint4 *) (tb + ((int64_t) sb*64 + lane)*16) = make_uint4(w[0], w[1], w[2], w[3]);
    }
    if (lane < 16) {
        const block_q2_K * b = src + (int64_t) (t*16 + lane)*nsb + sb;
        *(uint4 *) (tb + (int64_t) nsb*1024 + ((int64_t) sb*16 + lane)*16) =
            make_uint4(qmma_ld32(b->scales), qmma_ld32(b->scales + 4), qmma_ld32(b->scales + 8), qmma_ld32(b->scales + 12));
        *(half2 *) (tb + (int64_t) nsb*1280 + ((int64_t) sb*16 + lane)*4) = b->dm;
    }
}
__global__ void qmma_repack_q6_K(const block_q6_K * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*64) {
        return;
    }
    const int lane = i % 64, sb = (i / 64) % nsb, t = i / (64*nsb);
    const int c = lane % 4;
    uint8_t * tb = dst + (int64_t) t*16*nsb*sizeof(block_q6_K);
    {
        const block_q6_K * b = src + (int64_t) (t*16 + lane/4)*nsb + sb;
        uint32_t ql[8], qh[4];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int part = 0; part < 2; ++part) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    ql[n*4 + part*2 + h] = qmma_ld32(b->ql + 64*n + 32*part + 16*h + 4*c);
                }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                qh[n*2 + h] = qmma_ld32(b->qh + 32*n + 16*h + 4*c);
            }
        }
        uint4 * o = (uint4 *) (tb + ((int64_t) sb*64 + lane)*32);
        o[0] = make_uint4(ql[0], ql[1], ql[2], ql[3]);
        o[1] = make_uint4(ql[4], ql[5], ql[6], ql[7]);
        *(uint4 *) (tb + (int64_t) nsb*2048 + ((int64_t) sb*64 + lane)*16) = make_uint4(qh[0], qh[1], qh[2], qh[3]);
    }
    if (lane < 16) {
        const block_q6_K * b = src + (int64_t) (t*16 + lane)*nsb + sb;
        const uint8_t * sc = (const uint8_t *) b->scales;
        *(uint4 *) (tb + (int64_t) nsb*3072 + ((int64_t) sb*16 + lane)*16) = make_uint4(qmma_ld32(sc), qmma_ld32(sc + 4), qmma_ld32(sc + 8), qmma_ld32(sc + 12));
        *(half *) (tb + (int64_t) nsb*3328 + ((int64_t) sb*16 + lane)*2) = b->d;
    }
}
__global__ void qmma_repack_iq2_xxs(const block_iq2_xxs * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*16) {
        return;
    }
    const int r = i % 16, sb = (i / 16) % nsb, t = i / (16*nsb);
    const block_iq2_xxs * b = src + (int64_t) (t*16 + r)*nsb + sb;
    uint8_t * tb = dst + (int64_t) t*16*nsb*sizeof(block_iq2_xxs);
    uint32_t * ix = (uint32_t *) (tb + ((int64_t) sb*16 + r)*32);
    uint32_t * ax = (uint32_t *) (tb + (int64_t) nsb*512 + ((int64_t) sb*16 + r)*32);
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        ix[ib] = b->qs[4*ib + 0] | ((uint32_t) b->qs[4*ib + 1] << 16);
        ax[ib] = b->qs[4*ib + 2] | ((uint32_t) b->qs[4*ib + 3] << 16);
    }
    *(half *) (tb + (int64_t) nsb*1024 + ((int64_t) sb*16 + r)*2) = b->d;
}
__global__ void qmma_repack_iq2_xs(const block_iq2_xs * __restrict__ src, uint8_t * __restrict__ dst, const int ntile, const int nsb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ntile*nsb*16) {
        return;
    }
    const int r = i % 16, sb = (i / 16) % nsb, t = i / (16*nsb);
    const block_iq2_xs * b = src + (int64_t) (t*16 + r)*nsb + sb;
    uint8_t * tb = dst + (int64_t) t*16*nsb*sizeof(block_iq2_xs);
#pragma unroll
    for (int cp = 0; cp < 2; ++cp) {
        uint32_t w[8];
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            w[ib] = b->qs[4*ib + cp] | ((uint32_t) b->qs[4*ib + 2 + cp] << 16);   // items (ib,h=0),(ib,h=1)
        }
        uint4 * o = (uint4 *) (tb + (((int64_t) sb*16 + r)*2 + cp)*32);
        o[0] = make_uint4(w[0], w[1], w[2], w[3]);
        o[1] = make_uint4(w[4], w[5], w[6], w[7]);
    }
    *(uint2 *) (tb + (int64_t) nsb*1024 + ((int64_t) sb*16 + r)*8) = make_uint2(qmma_ld32(b->scales), qmma_ld32(b->scales + 4));
    *(half *) (tb + (int64_t) nsb*1152 + ((int64_t) sb*16 + r)*2) = b->d;
}

// =============================================================================================
// activation quantizer: 8 threads per (token, 32-block), 4 values each; reductions through shared memory
// (the SDK's __shfl is a lock + shared round trip on mp_21). Also writes the sum of the first 16 values
// of each block (dl), needed by types with per-16 scales.
#define QMMA_QBLK 16    // 32-blocks per 128-thread CTA
__global__ void qmma_quantize(const float * __restrict__ x, const int64_t stride_tok, uint32_t * __restrict__ ya, float2 * __restrict__ da,
                              float * __restrict__ dl, const int nb, const int ntok) {
    __shared__ float s_amax[QMMA_QBLK][8], s_sum[QMMA_QBLK][8];
    const int lb = threadIdx.x >> 3;               // block within CTA
    const int i  = threadIdx.x & 7;                // 4-value chunk within the 32-block
    const int b  = blockIdx.x*QMMA_QBLK + lb;
    const int t  = blockIdx.y;
    const bool valid = b < nb && t < ntok;
    float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (valid) {
        f = *(const float4 *) (x + t*stride_tok + (int64_t) b*32 + 4*i);
    }
    s_amax[lb][i] = fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w)));
    __syncthreads();
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        amax = fmaxf(amax, s_amax[lb][k]);
    }
    const float d  = amax / 127.0f;
    const float id = amax == 0.0f ? 0.0f : 1.0f/d;
    const int q0 = (int) roundf(f.x*id), q1 = (int) roundf(f.y*id), q2 = (int) roundf(f.z*id), q3 = (int) roundf(f.w*id);
    const uint32_t u = (uint32_t) (uint8_t) (int8_t) q0 | ((uint32_t) (uint8_t) (int8_t) q1 << 8)
                     | ((uint32_t) (uint8_t) (int8_t) q2 << 16) | ((uint32_t) (uint8_t) (int8_t) q3 << 24);
    // exact sums of the QUANTIZED values: the per-type constant offsets (+128 etc.) must cancel exactly
    s_sum[lb][i] = (float) (q0 + q1 + q2 + q3);
    __syncthreads();
    if (!valid) {
        return;
    }
    float sum = 0.0f, sumlo = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        sum += s_sum[lb][k];
        if (k < 4) {
            sumlo += s_sum[lb][k];
        }
    }
    sum *= d;
    sumlo *= d;
    const int ngrp = nb/4, g = b/4, j = b%4;
    uint32_t * o = ya + ((int64_t) t*ngrp + g)*32;
    // chunk i < 4: y[4i..4i+3] -> c = i, slot 2j ; chunk i >= 4: y[16+4(i-4)..] -> c = i-4, slot 2j+1
    o[(i & 3)*8 + 2*j + (i >> 2)] = u ^ 0x80808080u;
    if (i == 0) {
        da[((int64_t) t*ngrp + g)*4 + j] = make_float2(d, sum);
        dl[((int64_t) t*ngrp + g)*4 + j] = sumlo;
    }
}

// =============================================================================================
// MMA helpers shared by the kernels
struct qmma_act {
    uint32_t yv[8];
    float    dsx[4], dsy[4];
};

__device__ __forceinline__ void qmma_load_act(qmma_act & a, const uint4 * __restrict__ yp, const float4 * __restrict__ dap,
                                              const int g, const int l, const int bt, const int N, const int ones) {
    uint4 ylo = make_uint4(0, 0, 0, 0), yhi = make_uint4(0, 0, 0, 0);
    if (l < 32) {
        if (bt < N) {
            ylo = yp[(int64_t) g*8];
            yhi = yp[(int64_t) g*8 + 1];
        } else if (bt == 7) {
            ylo = make_uint4(ones, ones, ones, ones);
            yhi = ylo;
        }
    }
    const float4 ds01 = dap[(int64_t) g*2], ds23 = dap[(int64_t) g*2 + 1];
    a.dsx[0] = ds01.x; a.dsx[1] = ds01.z; a.dsx[2] = ds23.x; a.dsx[3] = ds23.z;
    a.dsy[0] = ds01.y; a.dsy[1] = ds01.w; a.dsy[2] = ds23.y; a.dsy[3] = ds23.w;
    a.yv[0] = ylo.x; a.yv[1] = ylo.y; a.yv[2] = ylo.z; a.yv[3] = ylo.w;
    a.yv[4] = yhi.x; a.yv[5] = yhi.y; a.yv[6] = yhi.z; a.yv[7] = yhi.w;
}

__device__ __forceinline__ void qmma_load_act_ldg(qmma_act & a, const uint4 * __restrict__ yp, const float4 * __restrict__ dap,
                                                  const int g, const int l, const int bt, const int N, const int ones) {
    uint4 ylo = make_uint4(0, 0, 0, 0), yhi = make_uint4(0, 0, 0, 0);
    if (l < 32) {
        if (bt < N) {
            ylo = __ldg(yp + (int64_t) g*8);
            yhi = __ldg(yp + (int64_t) g*8 + 1);
        } else if (bt == 7) {
            ylo = make_uint4(ones, ones, ones, ones);
            yhi = ylo;
        }
    }
    const float4 ds01 = __ldg(dap + (int64_t) g*2), ds23 = __ldg(dap + (int64_t) g*2 + 1);
    a.dsx[0] = ds01.x; a.dsx[1] = ds01.z; a.dsx[2] = ds23.x; a.dsx[3] = ds23.z;
    a.dsy[0] = ds01.y; a.dsy[1] = ds01.w; a.dsy[2] = ds23.y; a.dsy[3] = ds23.w;
    a.yv[0] = ylo.x; a.yv[1] = ylo.y; a.yv[2] = ylo.z; a.yv[3] = ylo.w;
    a.yv[4] = yhi.x; a.yv[5] = yhi.y; a.yv[6] = yhi.z; a.yv[7] = yhi.w;
}

// two chained MMAs over one 32-k block (A halves a0/a1, B halves b0/b1); returns D for this lane
__device__ __forceinline__ int qmma_block(const uint32_t a0v, const uint32_t a1v, const uint32_t b0v, const uint32_t b1v) {
    int a0[2] = {(int) a0v, 0}, a1[2] = {(int) a1v, 0};
    int b0[1] = {(int) b0v}, b1[1] = {(int) b1v};
    int c[4] = {0, 0, 0, 0}, d[4] = {0, 0, 0, 0};
    __musa_imma_m16n8k16_mma(d, a0, b0, c, 0, 0, 0, false, 0, 1);
    c[0] = d[0];
    __musa_imma_m16n8k16_mma(d, a1, b1, c, 0, 0, 0, false, 0, 1);
    return d[0];
}

// =============================================================================================
// ---- split-K for narrow matrices (few 16-row tiles): ks warps of a block share one tile, each walks every ks-th
// superblock, partial sums are combined through shared memory (no atomics: any atomicAdd makes mcc 4.2 kernels ~6x slower)
#define QMMA_LB(KS, B) ((((KS) == 1) ? (B) : 4)*128)
#define QMMA_KS_TILE(W)                                                                                      \
    const int kp = wp % ks;                                                                                  \
    const int tile = blockIdx.x*(nwk/ks) + wp/ks;                                                            \
    const bool act = tile*16 < R;
#define QMMA_KS_STORE(W)                                                                                     \
    {                                                                                                        \
        __shared__ float qks_red[(KS > 1) ? nwk : 1][128];                                                   \
        if (ks > 1) {                                                                                        \
            qks_red[wp][l] = acc;                                                                            \
            __syncthreads();                                                                                 \
            if (kp == 0) {                                                                                   \
                for (int q = 1; q < ks; ++q) {                                                               \
                    acc += qks_red[wp + q][l];                                                               \
                }                                                                                            \
            }                                                                                                \
        }                                                                                                    \
        const int r = tile*16 + orow;                                                                        \
        if (kp == 0 && act && ocol < N && r < R) {                                                           \
            dst[ocol*stride_dst_tok + r] = acc;                                                              \
        }                                                                                                    \
    }

// IQ4_XS: 2 warps per block share the 1 KB pair table (T[b] = (kv[b&15]+128) | (kv[b>>4]+128) << 16)
#define QMMA_IQ4_WARPS 4
template <int KS>
__launch_bounds__(QMMA_LB(KS, 2), 1)
__global__ void qmma_iq4_xs(const uint8_t * __restrict__ w, const uint4 * __restrict__ ya, const float4 * __restrict__ da,
                            float * __restrict__ dst, const int64_t stride_dst_tok, const int R, const int nsb, const int N, const int ones) {
    constexpr int ks = KS, nwk = (KS == 1) ? 2 : 4;
    __shared__ uint32_t ptab[256];
    __shared__ int slot[nwk][16];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        ptab[i] = (uint32_t) (uint8_t) (kvalues_iq4nl[i & 15] + 128) | ((uint32_t) (uint8_t) (kvalues_iq4nl[i >> 4] + 128) << 16);
    }
    __syncthreads();
    const int l = threadIdx.x & 127, wp = threadIdx.x >> 7;
    QMMA_KS_TILE(QMMA_IQ4_WARPS)
    const int ngrp = nsb*2;
    const int bt = l >> 2, orow = l >> 3, ocol = l & 7, L = l & 63;
    const uint8_t * tb = w + (int64_t) tile*16*nsb*sizeof(block_iq4_xs);
    const uint4 * yp = ya + ((int64_t) min(bt, N-1)*ngrp)*8 + (l & 3)*2;
    const float4 * dap = da + (int64_t) min(ocol, N-1)*ngrp*2;
    float acc = 0.0f;
    for (int sb = kp; act && sb < nsb; sb += ks) {
        const uint2 scv = ((const uint2 *) (tb + (int64_t) nsb*2048))[sb*16 + orow];
        const uint64_t scq = (uint64_t) scv.x | ((uint64_t) scv.y << 32);
        const float dsb = __half2float(__ushort_as_half((unsigned short) (scq & 0xFFFF)));
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            qmma_act a;
            qmma_load_act(a, yp, dap, 2*sb + gg, l, bt, N, ones);
            uint4 q = make_uint4(0, 0, 0, 0);
            if (l < 64) {
                q = ((const uint4 *) tb)[((int64_t) sb*64 + L)*2 + gg];
            }
            const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ib = 4*gg + j;
                const uint32_t v = qw[j];
                const uint32_t P = ptab[v & 0xFF] | (ptab[(v >> 8) & 0xFF] << 8);
                const uint32_t Y = ptab[(v >> 16) & 0xFF] | (ptab[v >> 24] << 8);
                const int D = qmma_block((P & 0xFFFF) | (Y << 16), (P >> 16) | (Y & 0xFFFF0000u), a.yv[2*j], a.yv[2*j + 1]);
                if (ocol == 7) {
                    slot[wp][orow] = D;
                }
                __musa_syncwarp();
                const int sa = slot[wp][orow];
                const float dws = dsb * (float) ((int) ((scq >> (16 + 6*ib)) & 63) - 32);
                acc += dws * (a.dsx[j]*(float) (D - 128*sa) - 128.0f*a.dsy[j]);
            }
        }
    }
    QMMA_KS_STORE(QMMA_IQ4_WARPS)
}

// Q8_0: int8 values, one scale per 32: A' = q ^ 0x80 (= q + 128), no tables. Row sums exchanged once per superblock,
// activations of both halves loaded at the top of the superblock (as Q4_0). V2=false: the original per-block exchange.
#define QMMA_Q8_WARPS 4
template <int KS, bool V2 = true>
__launch_bounds__(QMMA_LB(KS, 2), 1)
__global__ void qmma_q8_0(const uint8_t * __restrict__ w, const uint4 * __restrict__ ya, const float4 * __restrict__ da,
                          float * __restrict__ dst, const int64_t stride_dst_tok, const int R, const int nsb, const int N, const int ones) {
    constexpr int ks = KS, nwk = (KS == 1) ? 2 : 4;
    __shared__ int slot[nwk][16];
    __shared__ int4 ssum[V2 ? nwk : 1][16][2];
    const int l = threadIdx.x & 127, wp = threadIdx.x >> 7;
    QMMA_KS_TILE(QMMA_Q8_WARPS)
    const int ngrp = nsb*2;
    const int bt = l >> 2, orow = l >> 3, ocol = l & 7, L = l & 63;
    const uint8_t * tb = w + (int64_t) tile*16*nsb*8*sizeof(block_q8_0);
    const uint4 * yp = ya + ((int64_t) min(bt, N-1)*ngrp)*8 + (l & 3)*2;
    const float4 * dap = da + (int64_t) min(ocol, N-1)*ngrp*2;
    float acc = 0.0f;
    for (int sb = kp; act && sb < nsb; sb += ks) {
        qmma_act act2[2];
        if constexpr (V2) {
            qmma_load_act_ldg(act2[0], yp, dap, 2*sb,     l, bt, N, ones);
            qmma_load_act_ldg(act2[1], yp, dap, 2*sb + 1, l, bt, N, ones);
        }
        const uint4 dv = ((const uint4 *) (tb + (int64_t) nsb*4096))[sb*16 + orow];
        const uint32_t dw[4] = {dv.x, dv.y, dv.z, dv.w};
        int sv[8];
        float wdx[8];
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            qmma_act a;
            if constexpr (V2) {
                a = act2[gg];
            } else {
                qmma_load_act(a, yp, dap, 2*sb + gg, l, bt, N, ones);
            }
            uint4 q0 = make_uint4(0, 0, 0, 0), q1 = q0;
            if (l < 64) {
                q0 = ((const uint4 *) tb)[(((int64_t) sb*64 + L)*2 + gg)*2];
                q1 = ((const uint4 *) tb)[(((int64_t) sb*64 + L)*2 + gg)*2 + 1];
            }
            const uint32_t qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ib = 4*gg + j;
                const int D = qmma_block(qw[2*j] ^ 0x80808080u, qw[2*j + 1] ^ 0x80808080u, a.yv[2*j], a.yv[2*j + 1]);
                const float d = __half2float(__ushort_as_half((unsigned short) (dw[ib >> 1] >> (16*(ib & 1)))));
                if constexpr (V2) {
                    wdx[ib] = d*a.dsx[j];
                    sv[ib]  = D;
                    acc = fmaf(wdx[ib], (float) D, fmaf(-128.0f*d, a.dsy[j], acc));
                } else {
                    if (ocol == 7) {
                        slot[wp][orow] = D;
                    }
                    __musa_syncwarp();
                    const int sa = slot[wp][orow];
                    acc += d * (a.dsx[j]*(float) (D - 128*sa) - 128.0f*a.dsy[j]);
                }
            }
        }
        if constexpr (V2) {
            if (ocol == 7) {
                ssum[wp][orow][0] = make_int4(sv[0], sv[1], sv[2], sv[3]);
                ssum[wp][orow][1] = make_int4(sv[4], sv[5], sv[6], sv[7]);
            }
            __musa_syncwarp();
            const int4 s0 = ssum[wp][orow][0], s1 = ssum[wp][orow][1];
            acc -= 128.0f*(wdx[0]*(float) s0.x + wdx[1]*(float) s0.y + wdx[2]*(float) s0.z + wdx[3]*(float) s0.w +
                           wdx[4]*(float) s1.x + wdx[5]*(float) s1.y + wdx[6]*(float) s1.z + wdx[7]*(float) s1.w);
            __musa_syncwarp();
        }
    }
    QMMA_KS_STORE(QMMA_Q8_WARPS)
}

// Q4_0 / Q4_1: unsigned nibbles q (Q4_0: w = d*(q-8); Q4_1: w = d*q + m), 8 blocks per superblock.
// Row sums (ones column) of all 8 blocks are exchanged once per superblock (as in IQ3_S), not once per block, and both
// halves' activations are loaded at the top of the superblock (~5% over loading them per half).
#define QMMA_Q4X_WARPS 4
template <int KS, bool Q41>
__launch_bounds__(QMMA_LB(KS, 2), 1)
__global__ void qmma_q4_x(const uint8_t * __restrict__ w, const uint4 * __restrict__ ya, const float4 * __restrict__ da,
                          float * __restrict__ dst, const int64_t stride_dst_tok, const int R, const int nsb, const int N, const int ones) {
    constexpr int ks = KS, nwk = (KS == 1) ? 2 : 4;
    constexpr int BS = Q41 ? (int) sizeof(block_q4_1) : (int) sizeof(block_q4_0);
    __shared__ int4 ssum[nwk][16][2];
    const int l = threadIdx.x & 127, wp = threadIdx.x >> 7;
    QMMA_KS_TILE(QMMA_Q4X_WARPS)
    const int ngrp = nsb*2;
    const int bt = l >> 2, orow = l >> 3, ocol = l & 7, L = l & 63;
    const uint8_t * tb = w + (int64_t) tile*16*nsb*8*BS;
    const uint4 * yp = ya + ((int64_t) min(bt, N-1)*ngrp)*8 + (l & 3)*2;
    const float4 * dap = da + (int64_t) min(ocol, N-1)*ngrp*2;
    float acc = 0.0f;
    for (int sb = kp; act && sb < nsb; sb += ks) {
        qmma_act act2[2];
        qmma_load_act_ldg(act2[0], yp, dap, 2*sb,     l, bt, N, ones);
        qmma_load_act_ldg(act2[1], yp, dap, 2*sb + 1, l, bt, N, ones);
        uint32_t dw[8];
        if constexpr (Q41) {
            const uint4 d0 = ((const uint4 *) (tb + (int64_t) nsb*2048))[(sb*16 + orow)*2];
            const uint4 d1 = ((const uint4 *) (tb + (int64_t) nsb*2048))[(sb*16 + orow)*2 + 1];
            dw[0] = d0.x; dw[1] = d0.y; dw[2] = d0.z; dw[3] = d0.w; dw[4] = d1.x; dw[5] = d1.y; dw[6] = d1.z; dw[7] = d1.w;
        } else {
            const uint4 dv = ((const uint4 *) (tb + (int64_t) nsb*2048))[sb*16 + orow];
            dw[0] = dv.x; dw[1] = dv.y; dw[2] = dv.z; dw[3] = dv.w;
        }
        uint4 q0 = make_uint4(0, 0, 0, 0), q1 = q0;
        if (l < 64) {
            q0 = ((const uint4 *) tb)[((int64_t) sb*64 + L)*2];
            q1 = ((const uint4 *) tb)[((int64_t) sb*64 + L)*2 + 1];
        }
        const uint32_t qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        int sv[8];
        float wdx[8];
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            const qmma_act & a = act2[gg];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ib = 4*gg + j;
                const uint32_t v = qw[ib];
                const int D = qmma_block(v & 0x0F0F0F0F, (v >> 4) & 0x0F0F0F0F, a.yv[2*j], a.yv[2*j + 1]);
                float d, c;
                if constexpr (Q41) {
                    const float2 dm = __half22float2(*(const half2 *) &dw[ib]);
                    d = dm.x; c = dm.y;
                } else {
                    d = __half2float(__ushort_as_half((unsigned short) (dw[ib >> 1] >> (16*(ib & 1)))));
                    c = -8.0f*d;
                }
                wdx[ib] = d*a.dsx[j];
                sv[ib]  = D;
                acc = fmaf(wdx[ib], (float) D, fmaf(c, a.dsy[j], acc));
            }
        }
        if (ocol == 7) {
            ssum[wp][orow][0] = make_int4(sv[0], sv[1], sv[2], sv[3]);
            ssum[wp][orow][1] = make_int4(sv[4], sv[5], sv[6], sv[7]);
        }
        __musa_syncwarp();
        const int4 s0 = ssum[wp][orow][0], s1 = ssum[wp][orow][1];
        acc -= 128.0f*(wdx[0]*(float) s0.x + wdx[1]*(float) s0.y + wdx[2]*(float) s0.z + wdx[3]*(float) s0.w +
                       wdx[4]*(float) s1.x + wdx[5]*(float) s1.y + wdx[6]*(float) s1.z + wdx[7]*(float) s1.w);
        __musa_syncwarp();
    }
    QMMA_KS_STORE(QMMA_Q4X_WARPS)
}

// IQ3_S: 2 warps per block share the 2 KB grid copy; lanes 0-63 decode both halves (A' = 16 + sign*grid)
#define QMMA_IQ3_WARPS 4
template <int KS>
__launch_bounds__(QMMA_LB(KS, 2), 1)
__global__ void qmma_iq3_s(const uint8_t * __restrict__ w, const uint4 * __restrict__ ya, const float4 * __restrict__ da,
                           float * __restrict__ dst, const int64_t stride_dst_tok, const int R, const int nsb, const int N, const int ones) {
    constexpr int ks = KS, nwk = (KS == 1) ? 2 : 4;
    __shared__ uint32_t grid[512];
    __shared__ int4 ssum[nwk][16][2];
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        grid[i] = iq3s_grid[i];
    }
    __syncthreads();
    const int l = threadIdx.x & 127, wp = threadIdx.x >> 7;
    QMMA_KS_TILE(QMMA_IQ3_WARPS)
    const int ngrp = nsb*2;
    const int bt = l >> 2, orow = l >> 3, ocol = l & 7, L = l & 63;
    const uint8_t * tb = w + (int64_t) tile*16*nsb*sizeof(block_iq3_s);
    const uint4 * yp = ya + ((int64_t) min(bt, N-1)*ngrp)*8 + (l & 3)*2;
    const float4 * dap = da + (int64_t) min(ocol, N-1)*ngrp*2;
    float acc = 0.0f;
    for (int sb = kp; act && sb < nsb; sb += ks) {
        uint2 qs0 = make_uint2(0, 0), qs1 = make_uint2(0, 0), sgb = make_uint2(0, 0);
        uint32_t qh = 0;
        if (l < 64) {
            qs0 = ((const uint2 *) tb)[(sb*2 + 0)*64 + L];
            qs1 = ((const uint2 *) tb)[(sb*2 + 1)*64 + L];
            sgb = ((const uint2 *) (tb + iq3_o_sg(nsb)))[sb*64 + L];
            qh  = (uint32_t) ((const uint16_t *) (tb + iq3_o_qh(nsb)))[sb*64 + L] << 8;   // item k's 9th bit at bit 8+k
        }
        const float dsb = __half2float(((const half *) (tb + iq3_o_d(nsb)))[sb*16 + orow]);
        const uint32_t scw = ((const uint32_t *) (tb + iq3_o_sc(nsb)))[sb*16 + orow];
        int sv[8];
        float wdx[8];
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            qmma_act a;
            qmma_load_act(a, yp, dap, 2*sb + gg, l, bt, N, ones);
            const uint32_t q0 = gg ? qs0.y : qs0.x, q1 = gg ? qs1.y : qs1.x;
            const uint32_t sw = gg ? sgb.y : sgb.x;    // sign bits of items k = 8gg .. 8gg+7
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ib = 4*gg + j;
                uint32_t av[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int k = 2*ib + h;
                    const uint32_t qb  = ((h ? q1 : q0) >> (8*j)) & 0xFF;
                    const uint32_t gv  = grid[qb | ((qh >> k) & 0x100)];
                    const uint32_t m01 = (sw >> (k & 7)) & 0x01010101u;     // 1 in negative bytes
                    av[h] = (gv ^ (0x80808080u - m01)) + m01;               // 128 + sign*g, no carries (g <= 15)
                }
                const int D = qmma_block(av[0], av[1], a.yv[2*j], a.yv[2*j + 1]);
                const float dws = dsb * (float) (1 + 2*((scw >> (4*ib)) & 0xF));
                wdx[ib] = dws*a.dsx[j];
                sv[ib]  = D;
                acc += wdx[ib]*(float) D - 128.0f*dws*a.dsy[j];
            }
        }
        if (ocol == 7) {
            ssum[wp][orow][0] = make_int4(sv[0], sv[1], sv[2], sv[3]);
            ssum[wp][orow][1] = make_int4(sv[4], sv[5], sv[6], sv[7]);
        }
        __musa_syncwarp();
        const int4 s0 = ssum[wp][orow][0], s1 = ssum[wp][orow][1];
        acc -= 128.0f*(wdx[0]*(float) s0.x + wdx[1]*(float) s0.y + wdx[2]*(float) s0.z + wdx[3]*(float) s0.w +
                       wdx[4]*(float) s1.x + wdx[5]*(float) s1.y + wdx[6]*(float) s1.z + wdx[7]*(float) s1.w);
        __musa_syncwarp();
    }
    QMMA_KS_STORE(QMMA_IQ3_WARPS)
}

// IQ3_XXS: A' = 64 + sign*grid in [2,126]; 2 warps per block share the 1 KB grid copy
#define QMMA_IQ3XXS_WARPS 4
template <int KS>
__launch_bounds__(QMMA_LB(KS, 2), 1)
__global__ void qmma_iq3_xxs(const uint8_t * __restrict__ w, const uint4 * __restrict__ ya, const float4 * __restrict__ da,
                             float * __restrict__ dst, const int64_t stride_dst_tok, const int R, const int nsb, const int N, const int ones) {
    constexpr int ks = KS, nwk = (KS == 1) ? 2 : 4;
    __shared__ uint32_t grid[256];
    __shared__ uint32_t stab[256];     // [7 sign bits][nibble] -> byte-spread sign bits (8th sign = parity)
    __shared__ int slot[nwk][16];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        grid[i] = iq3xxs_grid[i];
        const uint32_t g7 = i >> 1, s8 = g7 | ((__popc(g7) & 1) << 7), nib = (s8 >> (4*(i & 1))) & 0xF;
        stab[i] = (nib * 0x00204081u) & 0x01010101u;
    }
    __syncthreads();
    const int l = threadIdx.x & 127, wp = threadIdx.x >> 7;
    QMMA_KS_TILE(QMMA_IQ3XXS_WARPS)
    const int ngrp = nsb*2;
    const int bt = l >> 2, orow = l >> 3, ocol = l & 7, L = l & 63, c = L & 3, arow = L >> 2;
    const int sh0 = 7*(c >> 1), sh1 = 7*(2 + (c >> 1)), ch = c & 1;
    const uint8_t * tb = w + (int64_t) tile*16*nsb*sizeof(block_iq3_xxs);
    const uint4 * auxp = (const uint4 *) (tb + (int64_t) nsb*1024);
    const uint4 * yp = ya + ((int64_t) min(bt, N-1)*ngrp)*8 + (l & 3)*2;
    const float4 * dap = da + (int64_t) min(ocol, N-1)*ngrp*2;
    float acc = 0.0f;
    for (int sb = kp; act && sb < nsb; sb += ks) {
        uint2 q0 = make_uint2(0, 0), q1 = make_uint2(0, 0);
        uint4 ax0 = make_uint4(0, 0, 0, 0), ax1 = make_uint4(0, 0, 0, 0);
        if (l < 64) {
            q0  = ((const uint2 *) tb)[(sb*2 + 0)*64 + L];
            q1  = ((const uint2 *) tb)[(sb*2 + 1)*64 + L];
            ax0 = auxp[(sb*16 + arow)*2];
            ax1 = auxp[(sb*16 + arow)*2 + 1];
        }
        const uint4 ox0 = auxp[(sb*16 + orow)*2], ox1 = auxp[(sb*16 + orow)*2 + 1];
        const float dsb = __half2float(((const half *) (tb + (int64_t) nsb*1536))[sb*16 + orow]);
        const uint32_t ax[8] = {ax0.x, ax0.y, ax0.z, ax0.w, ax1.x, ax1.y, ax1.z, ax1.w};
        const uint32_t ox[8] = {ox0.x, ox0.y, ox0.z, ox0.w, ox1.x, ox1.y, ox1.z, ox1.w};
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            qmma_act a;
            qmma_load_act(a, yp, dap, 2*sb + gg, l, bt, N, ones);
            const uint32_t qa = gg ? q0.y : q0.x, qb = gg ? q1.y : q1.x;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ib = 4*gg + j;
                uint32_t av[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t gv  = grid[((h ? qb : qa) >> (8*j)) & 0xFF];
                    const uint32_t m01 = stab[(((ax[ib] >> (h ? sh1 : sh0)) & 0x7F) << 1) | ch];
                    av[h] = (gv ^ (0x80808080u - m01)) + m01;               // 128 + sign*g (g <= 62)
                }
                const int D = qmma_block(av[0], av[1], a.yv[2*j], a.yv[2*j + 1]);
                if (ocol == 7) {
                    slot[wp][orow] = D;
                }
                __musa_syncwarp();
                const int sa = slot[wp][orow];
                const float dws = dsb * (0.5f + (float) (ox[ib] >> 28)) * 0.5f;
                acc += dws * (a.dsx[j]*(float) (D - 128*sa) - 128.0f*a.dsy[j]);
            }
        }
    }
    QMMA_KS_STORE(QMMA_IQ3XXS_WARPS)
}

// Q4_K: unsigned nibbles; w = d*sc*q - dmin*m
__device__ __forceinline__ void qmma_q4k_scmin(const uint8_t * q, const int j, int & d, int & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#define QMMA_Q4K_WARPS 4
template <int KS>
__launch_bounds__(QMMA_LB(KS, 1), 1)
__global__ void qmma_q4_K(const uint8_t * __restrict__ w, const uint4 * __restrict__ ya, const float4 * __restrict__ da,
                          float * __restrict__ dst, const int64_t stride_dst_tok, const int R, const int nsb, const int N, const int ones) {
    constexpr int ks = KS, nwk = (KS == 1) ? 1 : 4;
    __shared__ int slot[nwk][16];
    const int l = threadIdx.x & 127, wp = threadIdx.x >> 7;
    QMMA_KS_TILE(QMMA_Q4K_WARPS)
    const int ngrp = nsb*2;
    const int bt = l >> 2, orow = l >> 3, ocol = l & 7, L = l & 63;
    const uint8_t * tb = w + (int64_t) tile*16*nsb*sizeof(block_q4_K);
    const uint4 * yp = ya + ((int64_t) min(bt, N-1)*ngrp)*8 + (l & 3)*2;
    const float4 * dap = da + (int64_t) min(ocol, N-1)*ngrp*2;
    float acc = 0.0f;
    for (int sb = kp; act && sb < nsb; sb += ks) {
        const uint4 sc4 = ((const uint4 *) (tb + (int64_t) nsb*2048))[sb*16 + orow];
        const float2 dm = __half22float2(*(const half2 *) &sc4.x);
        const uint32_t scw[3] = {sc4.y, sc4.z, sc4.w};
        const uint8_t * scb = (const uint8_t *) scw;
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            qmma_act a;
            qmma_load_act(a, yp, dap, 2*sb + gg, l, bt, N, ones);
            uint4 q = make_uint4(0, 0, 0, 0);
            if (l < 64) {
                q = ((const uint4 *) tb)[((int64_t) sb*64 + L)*2 + gg];
            }
            const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ib = 4*gg + j;
                const int pw = 2*(j >> 1);
                const uint32_t A0 = (ib & 1) ? (qw[pw]     >> 4) & 0x0F0F0F0F : qw[pw]     & 0x0F0F0F0F;
                const uint32_t A1 = (ib & 1) ? (qw[pw + 1] >> 4) & 0x0F0F0F0F : qw[pw + 1] & 0x0F0F0F0F;
                const int D = qmma_block(A0, A1, a.yv[2*j], a.yv[2*j + 1]);
                if (ocol == 7) {
                    slot[wp][orow] = D;
                }
                __musa_syncwarp();
                const int sa = slot[wp][orow];
                int sc, mn;
                qmma_q4k_scmin(scb, ib, sc, mn);
                acc += dm.x*(float) sc*a.dsx[j]*(float) (D - 128*sa) - dm.y*(float) mn*a.dsy[j];
            }
        }
    }
    QMMA_KS_STORE(QMMA_Q4K_WARPS)
}

// Q5_K: 5-bit unsigned (nibble | qh bit << 4); w = d*sc*q - dmin*m
template <int KS>
__launch_bounds__(QMMA_LB(KS, 1), 1)
__global__ void qmma_q5_K(const uint8_t * __restrict__ w, const uint4 * __restrict__ ya, const float4 * __restrict__ da,
                          float * __restrict__ dst, const int64_t stride_dst_tok, const int R, const int nsb, const int N, const int ones) {
    constexpr int ks = KS, nwk = (KS == 1) ? 1 : 4;
    __shared__ int slot[nwk][16];
    const int l = threadIdx.x & 127, wp = threadIdx.x >> 7;
    QMMA_KS_TILE(QMMA_Q4K_WARPS)
    const int ngrp = nsb*2;
    const int bt = l >> 2, orow = l >> 3, ocol = l & 7, L = l & 63;
    const uint8_t * tb = w + (int64_t) tile*16*nsb*sizeof(block_q5_K);
    const uint4 * yp = ya + ((int64_t) min(bt, N-1)*ngrp)*8 + (l & 3)*2;
    const float4 * dap = da + (int64_t) min(ocol, N-1)*ngrp*2;
    float acc = 0.0f;
    for (int sb = kp; act && sb < nsb; sb += ks) {
        const uint4 sc4 = ((const uint4 *) (tb + (int64_t) nsb*2560))[sb*16 + orow];
        const float2 dm = __half22float2(*(const half2 *) &sc4.x);
        const uint32_t scw[3] = {sc4.y, sc4.z, sc4.w};
        const uint8_t * scb = (const uint8_t *) scw;
        uint2 qh = make_uint2(0, 0);
        if (l < 64) {
            qh = ((const uint2 *) (tb + (int64_t) nsb*2048))[sb*64 + L];
        }
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            qmma_act a;
            qmma_load_act(a, yp, dap, 2*sb + gg, l, bt, N, ones);
            uint4 q = make_uint4(0, 0, 0, 0);
            if (l < 64) {
                q = ((const uint4 *) tb)[((int64_t) sb*64 + L)*2 + gg];
            }
            const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ib = 4*gg + j;
                const int pw = 2*(j >> 1);
                const uint32_t A0 = ((ib & 1) ? (qw[pw]     >> 4) & 0x0F0F0F0F : qw[pw]     & 0x0F0F0F0F) | (((qh.x >> ib) & 0x01010101u) << 4);
                const uint32_t A1 = ((ib & 1) ? (qw[pw + 1] >> 4) & 0x0F0F0F0F : qw[pw + 1] & 0x0F0F0F0F) | (((qh.y >> ib) & 0x01010101u) << 4);
                const int D = qmma_block(A0, A1, a.yv[2*j], a.yv[2*j + 1]);
                if (ocol == 7) {
                    slot[wp][orow] = D;
                }
                __musa_syncwarp();
                const int sa = slot[wp][orow];
                int sc, mn;
                qmma_q4k_scmin(scb, ib, sc, mn);
                acc += dm.x*(float) sc*a.dsx[j]*(float) (D - 128*sa) - dm.y*(float) mn*a.dsy[j];
            }
        }
    }
    QMMA_KS_STORE(QMMA_Q4K_WARPS)
}

// IQ2_S: per-16 scales -> two unchained MMAs per 32-block, A' = 64 + sign*grid (grid 8/25/43).
// Register-heavy (~95), so occupancy is low: QMMA_IQ2S_SPLIT warps share one tile (split over superblocks)
// and the 8 KB grid sits in shared memory.
#define QMMA_IQ2S_SPLIT 8
__device__ __forceinline__ int qmma_one(const uint32_t av, const uint32_t bv) {
    int a[2] = {(int) av, 0}, b[1] = {(int) bv}, c[4] = {0, 0, 0, 0}, d[4] = {0, 0, 0, 0};
    __musa_imma_m16n8k16_mma(d, a, b, c, 0, 0, 0, false, 0, 1);
    return d[0];
}
__launch_bounds__(QMMA_IQ2S_SPLIT*128, 1)
__global__ void qmma_iq2_s(const uint8_t * __restrict__ w, const uint4 * __restrict__ ya, const float4 * __restrict__ da, const float4 * __restrict__ dl,
                           float * __restrict__ dst, const int64_t stride_dst_tok, const int R, const int nsb, const int N, const int ones) {
    __shared__ uint32_t grid[2048];
    __shared__ int2  slot[QMMA_IQ2S_SPLIT][16];
    __shared__ float red[QMMA_IQ2S_SPLIT][128];
    for (int i = threadIdx.x; i < 2048; i += QMMA_IQ2S_SPLIT*128) {
        grid[i] = ((const uint32_t *) iq2s_grid)[i];
    }
    __syncthreads();
    const int l = threadIdx.x & 127, wp = threadIdx.x >> 7, tile = blockIdx.x;
    const int ngrp = nsb*2;
    const int bt = l >> 2, orow = l >> 3, ocol = l & 7, L = l & 63, c = L & 3, cp = c >> 1, ch = c & 1, arow = L >> 2;
    const uint8_t * tb = w + (int64_t) tile*16*nsb*sizeof(block_iq2_s);
    const uint4 * yp = ya + ((int64_t) min(bt, N-1)*ngrp)*8 + (l & 3)*2;
    const float4 * dap = da + (int64_t) min(ocol, N-1)*ngrp*2;
    const float4 * dlp = dl + (int64_t) min(ocol, N-1)*ngrp;
    float acc = 0.0f;
    for (int sb = wp; sb < nsb; sb += QMMA_IQ2S_SPLIT) {
        uint4 idx = make_uint4(0, 0, 0, 0), sgn = make_uint4(0, 0, 0, 0);
        uint32_t qh = 0;
        if (l < 64) {
            idx = ((const uint4 *) tb)[(sb*16 + arow)*2 + cp];
            sgn = ((const uint4 *) (tb + (int64_t) nsb*512))[(sb*16 + arow)*2 + cp];
            qh  = ((const uint32_t *) (tb + (int64_t) nsb*1024))[(sb*16 + arow)*2 + cp];
        }
        const uint2 sc2 = ((const uint2 *) (tb + (int64_t) nsb*1152))[sb*16 + orow];
        const float dsb = __half2float(((const half *) (tb + (int64_t) nsb*1280))[sb*16 + orow]);
        const uint32_t iw[4] = {idx.x, idx.y, idx.z, idx.w}, sw[4] = {sgn.x, sgn.y, sgn.z, sgn.w};
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            qmma_act a;
            qmma_load_act(a, yp, dap, 2*sb + gg, l, bt, N, ones);
            const float4 lo4 = dlp[2*sb + gg];
            const float dlo[4] = {lo4.x, lo4.y, lo4.z, lo4.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ib = 4*gg + j;
                uint32_t av[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int bi = 2*ib + h;
                    const uint32_t ib8 = (iw[bi >> 2] >> (8*(bi & 3))) & 0xFF;
                    const uint32_t gv  = grid[2*(ib8 | (((qh >> (2*bi)) & 3) << 8)) + ch];
                    const uint32_t nib = (sw[bi >> 2] >> (8*(bi & 3) + 4*ch)) & 0xF;
                    const uint32_t m01 = (nib * 0x00204081u) & 0x01010101u;
                    av[h] = ((gv + 0x40404040u) ^ (m01 * 0x7Fu)) + m01;
                }
                const int D0 = qmma_one(av[0], a.yv[2*j]), D1 = qmma_one(av[1], a.yv[2*j + 1]);
                if (ocol == 7) {
                    slot[wp][orow] = make_int2(D0, D1);
                }
                __musa_syncwarp();
                const int2 S = slot[wp][orow];
                const uint32_t scb = ((ib < 4 ? sc2.x : sc2.y) >> (8*(ib & 3))) & 0xFF;
                const float dw0 = dsb*(0.5f + (float) (scb & 0xF))*0.25f, dw1 = dsb*(0.5f + (float) (scb >> 4))*0.25f;
                acc += dw0*(a.dsx[j]*(float) (D0 - 128*S.x) - 64.0f*dlo[j]) + dw1*(a.dsx[j]*(float) (D1 - 128*S.y) - 64.0f*(a.dsy[j] - dlo[j]));
            }
        }
    }
    red[wp][l] = acc;
    __syncthreads();
    if (wp == 0) {
        float v = 0.0f;
#pragma unroll
        for (int i = 0; i < QMMA_IQ2S_SPLIT; ++i) {
            v += red[i][l];
        }
        const int r = tile*16 + orow;
        if (ocol < N && r < R) {
            dst[ocol*stride_dst_tok + r] = v;
        }
    }
}

// ---- small types (kernels share this prologue)
#define QMMA_SMALL_WARPS 4
#define QMMA_SMALL_PRE(TYPE)                                                                                  \
    const int l = threadIdx.x & 127, wp = threadIdx.x >> 7;                                                  \
    QMMA_KS_TILE(QMMA_SMALL_WARPS)                                                                           \
    const int ngrp = nsb*2;                                                                                  \
    const int bt = l >> 2, orow = l >> 3, ocol = l & 7, L = l & 63, c = L & 3, arow = L >> 2;                \
    GGML_UNUSED(c); GGML_UNUSED(arow);                                                                       \
    const uint8_t * tb = w + (int64_t) tile*16*nsb*sizeof(TYPE);                                             \
    const uint4 * yp = ya + ((int64_t) min(bt, N-1)*ngrp)*8 + (l & 3)*2;                                     \
    const float4 * dap = da + (int64_t) min(ocol, N-1)*ngrp*2;                                               \
    const float4 * dlp = dl + (int64_t) min(ocol, N-1)*ngrp;                                                 \
    float acc = 0.0f;
#define QMMA_SMALL_POST QMMA_KS_STORE(QMMA_SMALL_WARPS)
#define QMMA_SMALL_ARGS const uint8_t * __restrict__ w, const uint4 * __restrict__ ya, const float4 * __restrict__ da, \
    const float4 * __restrict__ dl, float * __restrict__ dst, const int64_t stride_dst_tok, const int R, const int nsb, const int N, const int ones
#define QMMA_STAB_INIT                                                                                       \
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {                                          \
        const uint32_t g7 = i >> 1, s8 = g7 | ((__popc(g7) & 1) << 7), nb4 = (s8 >> (4*(i & 1))) & 0xF;      \
        stab[i] = (nb4 * 0x00204081u) & 0x01010101u;                                                         \
    }

template <int KS>
__launch_bounds__(QMMA_LB(KS, 2), 1)
__global__ void qmma_q2_K(QMMA_SMALL_ARGS) {
    constexpr int ks = KS, nwk = (KS == 1) ? 2 : 4;
    __shared__ int2 slot[nwk][16];
    QMMA_SMALL_PRE(block_q2_K)
    for (int sb = kp; act && sb < nsb; sb += ks) {
        uint4 qw4 = make_uint4(0, 0, 0, 0);
        if (l < 64) {
            qw4 = ((const uint4 *) tb)[sb*64 + L];
        }
        const uint4 sc4 = ((const uint4 *) (tb + (int64_t) nsb*1024))[sb*16 + orow];
        const float2 dm = __half22float2(*(const half2 *) (tb + (int64_t) nsb*1280 + ((int64_t) sb*16 + orow)*4));
        const uint32_t qw[4] = {qw4.x, qw4.y, qw4.z, qw4.w};
        const uint32_t scw[4] = {sc4.x, sc4.y, sc4.z, sc4.w};
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            qmma_act a;
            qmma_load_act(a, yp, dap, 2*sb + gg, l, bt, N, ones);
            const float4 lo4 = dlp[2*sb + gg];
            const float dlo[4] = {lo4.x, lo4.y, lo4.z, lo4.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int D0 = qmma_one((qw[2*gg]     >> (2*j)) & 0x03030303u, a.yv[2*j]);
                const int D1 = qmma_one((qw[2*gg + 1] >> (2*j)) & 0x03030303u, a.yv[2*j + 1]);
                if (ocol == 7) {
                    slot[wp][orow] = make_int2(D0, D1);
                }
                __musa_syncwarp();
                const int2 S = slot[wp][orow];
                const int is = 8*gg + 2*j;
                const uint32_t s0 = (scw[is >> 2] >> (8*(is & 3))) & 0xFF, s1 = (scw[(is + 1) >> 2] >> (8*((is + 1) & 3))) & 0xFF;
                acc += dm.x*a.dsx[j]*((float) (s0 & 0xF)*(float) (D0 - 128*S.x) + (float) (s1 & 0xF)*(float) (D1 - 128*S.y))
                     - dm.y*((float) (s0 >> 4)*dlo[j] + (float) (s1 >> 4)*(a.dsy[j] - dlo[j]));
            }
        }
    }
    QMMA_SMALL_POST
}

template <int KS>
__launch_bounds__(QMMA_LB(KS, 2), 1)
__global__ void qmma_q6_K(QMMA_SMALL_ARGS) {
    constexpr int ks = KS, nwk = (KS == 1) ? 2 : 4;
    __shared__ int2 slot[nwk][16];
    QMMA_SMALL_PRE(block_q6_K)
    for (int sb = kp; act && sb < nsb; sb += ks) {
        const uint4 sc4 = ((const uint4 *) (tb + (int64_t) nsb*3072))[sb*16 + orow];
        const float dsb = __half2float(((const half *) (tb + (int64_t) nsb*3328))[sb*16 + orow]);
        const int8_t * scb = (const int8_t *) &sc4;
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            uint4 ql = make_uint4(0, 0, 0, 0);
            uint2 qh = make_uint2(0, 0);
            if (l < 64) {
                ql = ((const uint4 *) tb)[(sb*64 + L)*2 + gg];
                qh = ((const uint2 *) (tb + (int64_t) nsb*2048))[(sb*64 + L)*2 + gg];
            }
            const uint32_t qlw[4] = {ql.x, ql.y, ql.z, ql.w};
            qmma_act a;
            qmma_load_act(a, yp, dap, 2*sb + gg, l, bt, N, ones);
            const float4 lo4 = dlp[2*sb + gg];
            const float dlo[4] = {lo4.x, lo4.y, lo4.z, lo4.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int part = j & 1, nsh = 4*(j >> 1);
                const uint32_t A0 = ((qlw[part*2]     >> nsh) & 0x0F0F0F0Fu) | (((qh.x >> (2*j)) & 0x03030303u) << 4);
                const uint32_t A1 = ((qlw[part*2 + 1] >> nsh) & 0x0F0F0F0Fu) | (((qh.y >> (2*j)) & 0x03030303u) << 4);
                const int D0 = qmma_one(A0, a.yv[2*j]), D1 = qmma_one(A1, a.yv[2*j + 1]);
                if (ocol == 7) {
                    slot[wp][orow] = make_int2(D0, D1);
                }
                __musa_syncwarp();
                const int2 S = slot[wp][orow];
                const float sc0 = dsb*(float) scb[8*gg + 2*j], sc1 = dsb*(float) scb[8*gg + 2*j + 1];
                acc += sc0*(a.dsx[j]*(float) (D0 - 128*S.x) - 32.0f*dlo[j]) + sc1*(a.dsx[j]*(float) (D1 - 128*S.y) - 32.0f*(a.dsy[j] - dlo[j]));
            }
        }
    }
    QMMA_SMALL_POST
}

template <int KS>
__launch_bounds__(QMMA_LB(KS, 2), 1)
__global__ void qmma_iq2_xxs(QMMA_SMALL_ARGS) {
    constexpr int ks = KS, nwk = (KS == 1) ? 2 : 4;
    __shared__ uint32_t grid[512];
    __shared__ uint32_t stab[256];
    __shared__ int slot[nwk][16];
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        grid[i] = ((const uint32_t *) iq2xxs_grid)[i];
    }
    QMMA_STAB_INIT
    __syncthreads();
    QMMA_SMALL_PRE(block_iq2_xxs)
    GGML_UNUSED(dlp);
    const int cp = c >> 1, ch = c & 1;
    const uint4 * ixp = (const uint4 *) tb;
    const uint4 * axp = (const uint4 *) (tb + (int64_t) nsb*512);
    for (int sb = kp; act && sb < nsb; sb += ks) {
        uint4 ix0 = make_uint4(0, 0, 0, 0), ix1 = ix0, ax0 = ix0, ax1 = ix0;
        if (l < 64) {
            ix0 = ixp[(sb*16 + arow)*2]; ix1 = ixp[(sb*16 + arow)*2 + 1];
            ax0 = axp[(sb*16 + arow)*2]; ax1 = axp[(sb*16 + arow)*2 + 1];
        }
        const uint4 ox0 = axp[(sb*16 + orow)*2], ox1 = axp[(sb*16 + orow)*2 + 1];
        const float dsb = __half2float(((const half *) (tb + (int64_t) nsb*1024))[sb*16 + orow]);
        const uint32_t ix[8] = {ix0.x, ix0.y, ix0.z, ix0.w, ix1.x, ix1.y, ix1.z, ix1.w};
        const uint32_t ax[8] = {ax0.x, ax0.y, ax0.z, ax0.w, ax1.x, ax1.y, ax1.z, ax1.w};
        const uint32_t ox[8] = {ox0.x, ox0.y, ox0.z, ox0.w, ox1.x, ox1.y, ox1.z, ox1.w};
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            qmma_act a;
            qmma_load_act(a, yp, dap, 2*sb + gg, l, bt, N, ones);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ib = 4*gg + j;
                uint32_t av[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int li = 2*h + cp;
                    const uint32_t gv  = grid[2*((ix[ib] >> (8*li)) & 0xFF) + ch];
                    const uint32_t m01 = stab[(((ax[ib] >> (7*li)) & 0x7F) << 1) | ch];
                    av[h] = (gv ^ (0x80808080u - m01)) + m01;
                }
                const int D = qmma_block(av[0], av[1], a.yv[2*j], a.yv[2*j + 1]);
                if (ocol == 7) {
                    slot[wp][orow] = D;
                }
                __musa_syncwarp();
                const int sa = slot[wp][orow];
                const float dws = dsb * (0.5f + (float) (ox[ib] >> 28)) * 0.25f;
                acc += dws * (a.dsx[j]*(float) (D - 128*sa) - 128.0f*a.dsy[j]);
            }
        }
    }
    QMMA_SMALL_POST
}

template <int KS>
__launch_bounds__(QMMA_LB(KS, 2), 1)
__global__ void qmma_iq2_xs(QMMA_SMALL_ARGS) {
    constexpr int ks = KS, nwk = (KS == 1) ? 2 : 4;
    __shared__ uint32_t grid[1024];
    __shared__ uint32_t stab[256];
    __shared__ int2 slot[nwk][16];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        grid[i] = ((const uint32_t *) iq2xs_grid)[i];
    }
    QMMA_STAB_INIT
    __syncthreads();
    QMMA_SMALL_PRE(block_iq2_xs)
    const int cp = c >> 1, ch = c & 1;
    for (int sb = kp; act && sb < nsb; sb += ks) {
        uint4 q0 = make_uint4(0, 0, 0, 0), q1 = q0;
        if (l < 64) {
            q0 = ((const uint4 *) tb)[((sb*16 + arow)*2 + cp)*2];
            q1 = ((const uint4 *) tb)[((sb*16 + arow)*2 + cp)*2 + 1];
        }
        const uint2 sc2 = ((const uint2 *) (tb + (int64_t) nsb*1024))[sb*16 + orow];
        const float dsb = __half2float(((const half *) (tb + (int64_t) nsb*1152))[sb*16 + orow]);
        const uint32_t qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            qmma_act a;
            qmma_load_act(a, yp, dap, 2*sb + gg, l, bt, N, ones);
            const float4 lo4 = dlp[2*sb + gg];
            const float dlo[4] = {lo4.x, lo4.y, lo4.z, lo4.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ib = 4*gg + j;
                uint32_t av[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t v   = (qw[ib] >> (16*h)) & 0xFFFF;
                    const uint32_t gv  = grid[2*(v & 511) + ch];
                    const uint32_t m01 = stab[((v >> 9) << 1) | ch];
                    av[h] = (gv ^ (0x80808080u - m01)) + m01;
                }
                const int D0 = qmma_one(av[0], a.yv[2*j]), D1 = qmma_one(av[1], a.yv[2*j + 1]);
                if (ocol == 7) {
                    slot[wp][orow] = make_int2(D0, D1);
                }
                __musa_syncwarp();
                const int2 S = slot[wp][orow];
                const uint32_t scb = ((ib < 4 ? sc2.x : sc2.y) >> (8*(ib & 3))) & 0xFF;
                const float dw0 = dsb*(0.5f + (float) (scb & 0xF))*0.25f, dw1 = dsb*(0.5f + (float) (scb >> 4))*0.25f;
                acc += dw0*(a.dsx[j]*(float) (D0 - 128*S.x) - 128.0f*dlo[j]) + dw1*(a.dsx[j]*(float) (D1 - 128*S.y) - 128.0f*(a.dsy[j] - dlo[j]));
            }
        }
    }
    QMMA_SMALL_POST
}

// =============================================================================================
// Large-N path (prompt processing): decode each 16x32 weight fragment ONCE and run it against up to
// QMMA_BIG_NG*8 tokens (8 per MMA B tile, all 8 columns used). Per-row sums S of A' come from one extra
// MMA against an all-ones B. Each type supplies a decoder: per superblock load(), per 32-block decode()
// into the two k16 A halves (lanes 0-63), and coef(): result per 16-half h = dx*f_h*(D_h - 128*S_h) + c_h*sum_h(dx*y).
#define QMMA_BIG_NG    8        // token groups of 8 per launch -> 64 tokens
#define QMMA_BIG_WARPS 2
#ifndef QMMA_BIG_W_IQ2S
#define QMMA_BIG_W_IQ2S 4
#endif
#ifndef QMMA_BIG_W_IQ2XS
#define QMMA_BIG_W_IQ2XS 2
#endif
#ifndef QMMA_BIG_PIPE
#define QMMA_BIG_PIPE 2   // token groups whose loads may be in flight together
#endif

// the big kernel strides tiles by 16*nsb*sizeof(blk) with nsb = K/256: one "block" must cover 256 values
struct qmma_q8_0_sb { block_q8_0 b[8]; };
static_assert(sizeof(qmma_q8_0_sb) == 8*sizeof(block_q8_0), "unexpected padding");
struct qmma_dec_q8_0 {
    using blk = qmma_q8_0_sb;
    static constexpr int WARPS = QMMA_BIG_WARPS;
    static constexpr bool PER16 = false;
    struct smem { int dummy; };
    static __device__ void init(smem & s) { GGML_UNUSED(s); }
    uint4 q[4]; uint32_t dw[4];
    __device__ void load(const uint8_t * tb, const int nsb, const int sb, const int l, const int orow) {
        const int L = l & 63;
        q[0] = q[1] = q[2] = q[3] = make_uint4(0, 0, 0, 0);
        if (l < 64) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                q[k] = ((const uint4 *) tb)[((int64_t) sb*64 + L)*4 + k];
            }
        }
        const uint4 dv = ((const uint4 *) (tb + (int64_t) nsb*4096))[sb*16 + orow];
        dw[0] = dv.x; dw[1] = dv.y; dw[2] = dv.z; dw[3] = dv.w;
    }
    __device__ void load_gg(const int gg) { GGML_UNUSED(gg); }
    __device__ void decode(const smem & s, const int gg, const int j, const int l, uint32_t & a0, uint32_t & a1) const {
        GGML_UNUSED(s); GGML_UNUSED(l);
        const uint4 & h = q[2*gg + (j >> 1)];
        const uint32_t w0 = (j & 1) ? h.z : h.x, w1 = (j & 1) ? h.w : h.y;
        a0 = w0 ^ 0x80808080u;
        a1 = w1 ^ 0x80808080u;
    }
    __device__ void coef(const int gg, const int j, float & f0, float & f1, float & c0, float & c1) const {
        const int ib = 4*gg + j;
        f0 = f1 = __half2float(__ushort_as_half((unsigned short) (dw[ib >> 1] >> (16*(ib & 1)))));
        c0 = c1 = -128.0f*f0;
    }
};

template <bool Q41> struct qmma_q4_x_sb { block_q4_0 b[8]; };
template <> struct qmma_q4_x_sb<true> { block_q4_1 b[8]; };
template <bool Q41>
struct qmma_dec_q4_x {
    using blk = qmma_q4_x_sb<Q41>;
    static constexpr int WARPS = QMMA_BIG_WARPS;
    static constexpr bool PER16 = false;
    struct smem { int dummy; };
    static __device__ void init(smem & s) { GGML_UNUSED(s); }
    uint4 q[2]; uint32_t dw[8];
    __device__ void load(const uint8_t * tb, const int nsb, const int sb, const int l, const int orow) {
        const int L = l & 63;
        q[0] = q[1] = make_uint4(0, 0, 0, 0);
        if (l < 64) {
            q[0] = ((const uint4 *) tb)[((int64_t) sb*64 + L)*2];
            q[1] = ((const uint4 *) tb)[((int64_t) sb*64 + L)*2 + 1];
        }
        if constexpr (Q41) {
            const uint4 d0 = ((const uint4 *) (tb + (int64_t) nsb*2048))[(sb*16 + orow)*2];
            const uint4 d1 = ((const uint4 *) (tb + (int64_t) nsb*2048))[(sb*16 + orow)*2 + 1];
            dw[0] = d0.x; dw[1] = d0.y; dw[2] = d0.z; dw[3] = d0.w;
            dw[4] = d1.x; dw[5] = d1.y; dw[6] = d1.z; dw[7] = d1.w;
        } else {
            const uint4 dv = ((const uint4 *) (tb + (int64_t) nsb*2048))[sb*16 + orow];
            dw[0] = dv.x; dw[1] = dv.y; dw[2] = dv.z; dw[3] = dv.w;
        }
    }
    __device__ void load_gg(const int gg) { GGML_UNUSED(gg); }
    __device__ void decode(const smem & s, const int gg, const int j, const int l, uint32_t & a0, uint32_t & a1) const {
        GGML_UNUSED(s); GGML_UNUSED(l);
        const uint4 & h = q[gg];
        const uint32_t v = j == 0 ? h.x : j == 1 ? h.y : j == 2 ? h.z : h.w;
        a0 = v & 0x0F0F0F0F;
        a1 = (v >> 4) & 0x0F0F0F0F;
    }
    __device__ void coef(const int gg, const int j, float & f0, float & f1, float & c0, float & c1) const {
        const int ib = 4*gg + j;
        if constexpr (Q41) {
            const float2 dm = __half22float2(*(const half2 *) &dw[ib]);
            f0 = f1 = dm.x;
            c0 = c1 = dm.y;
        } else {
            f0 = f1 = __half2float(__ushort_as_half((unsigned short) (dw[ib >> 1] >> (16*(ib & 1)))));
            c0 = c1 = -8.0f*f0;
        }
    }
};
using qmma_dec_q4_0 = qmma_dec_q4_x<false>;
using qmma_dec_q4_1 = qmma_dec_q4_x<true>;

struct qmma_dec_iq3_s {
    static constexpr int WARPS = QMMA_BIG_WARPS;
    using blk = block_iq3_s;
    static constexpr bool PER16 = false;
    struct smem { uint32_t grid[512]; };
    static __device__ void init(smem & s) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) s.grid[i] = iq3s_grid[i];
    }
    uint2 qs0, qs1, sgb; uint32_t qh; float dsb; uint32_t scw;
    __device__ void load(const uint8_t * tb, const int nsb, const int sb, const int l, const int orow) {
        const int L = l & 63;
        qs0 = qs1 = sgb = make_uint2(0, 0); qh = 0;
        if (l < 64) {
            qs0 = ((const uint2 *) tb)[(sb*2 + 0)*64 + L];
            qs1 = ((const uint2 *) tb)[(sb*2 + 1)*64 + L];
            sgb = ((const uint2 *) (tb + iq3_o_sg(nsb)))[sb*64 + L];
            qh  = (uint32_t) ((const uint16_t *) (tb + iq3_o_qh(nsb)))[sb*64 + L] << 8;
        }
        dsb = __half2float(((const half *) (tb + iq3_o_d(nsb)))[sb*16 + orow]);
        scw = ((const uint32_t *) (tb + iq3_o_sc(nsb)))[sb*16 + orow];
    }
    __device__ void load_gg(const int gg) { GGML_UNUSED(gg); }
    __device__ void decode(const smem & s, const int gg, const int j, const int l, uint32_t & a0, uint32_t & a1) const {
        GGML_UNUSED(l);
        const uint32_t q0 = gg ? qs0.y : qs0.x, q1 = gg ? qs1.y : qs1.x, sw = gg ? sgb.y : sgb.x;
        const int ib = 4*gg + j;
        uint32_t av[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int k = 2*ib + h;
            const uint32_t qb  = ((h ? q1 : q0) >> (8*j)) & 0xFF;
            const uint32_t gv  = s.grid[qb | ((qh >> k) & 0x100)];
            const uint32_t m01 = (sw >> (k & 7)) & 0x01010101u;
            av[h] = (gv ^ (0x80808080u - m01)) + m01;
        }
        a0 = av[0]; a1 = av[1];
    }
    __device__ void coef(const int gg, const int j, float & f0, float & f1, float & c0, float & c1) const {
        const int ib = 4*gg + j;
        f0 = f1 = dsb * (float) (1 + 2*((scw >> (4*ib)) & 0xF));
        c0 = c1 = -128.0f*f0;
    }
};

struct qmma_dec_iq3_xxs {
    static constexpr int WARPS = QMMA_BIG_WARPS;
    using blk = block_iq3_xxs;
    static constexpr bool PER16 = false;
    static constexpr int  OFF = 128;
    struct smem { uint32_t grid[256]; uint32_t stab[256]; };
    static __device__ void init(smem & s) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            s.grid[i] = iq3xxs_grid[i];
            const uint32_t g7 = i >> 1, s8 = g7 | ((__popc(g7) & 1) << 7), nib = (s8 >> (4*(i & 1))) & 0xF;
            s.stab[i] = (nib * 0x00204081u) & 0x01010101u;
        }
    }
    const uint8_t * tb; int nsb, sb, l, orow;
    uint2 q0, q1; uint4 ax, ox; float dsb;
    __device__ void load(const uint8_t * tb_, const int nsb_, const int sb_, const int l_, const int orow_) {
        tb = tb_; nsb = nsb_; sb = sb_; l = l_; orow = orow_;
        const int L = l & 63;
        q0 = q1 = make_uint2(0, 0);
        if (l < 64) {
            q0  = ((const uint2 *) tb)[(sb*2 + 0)*64 + L];
            q1  = ((const uint2 *) tb)[(sb*2 + 1)*64 + L];
        }
        dsb = __half2float(((const half *) (tb + (int64_t) nsb*1536))[sb*16 + orow]);
    }
    __device__ void load_gg(const int gg) {
        const uint4 * auxp = (const uint4 *) (tb + (int64_t) nsb*1024);
        const int arow = (l & 63) >> 2;
        ax = make_uint4(0, 0, 0, 0);
        if (l < 64) {
            ax = auxp[(sb*16 + arow)*2 + gg];
        }
        ox = auxp[(sb*16 + orow)*2 + gg];
    }
    __device__ void decode(const smem & s, const int gg, const int j, const int l_, uint32_t & a0, uint32_t & a1) const {
        GGML_UNUSED(l_);
        const int c = l & 3, sh0 = 7*(c >> 1), sh1 = 7*(2 + (c >> 1)), ch = c & 1;
        const uint32_t qa = gg ? q0.y : q0.x, qb = gg ? q1.y : q1.x;
        const uint32_t axj = j == 0 ? ax.x : j == 1 ? ax.y : j == 2 ? ax.z : ax.w;
        uint32_t av[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t gv  = s.grid[((h ? qb : qa) >> (8*j)) & 0xFF];
            const uint32_t m01 = s.stab[(((axj >> (h ? sh1 : sh0)) & 0x7F) << 1) | ch];
            av[h] = (gv ^ (0x80808080u - m01)) + m01;
        }
        a0 = av[0]; a1 = av[1];
    }
    __device__ void coef(const int gg, const int j, float & f0, float & f1, float & c0, float & c1) const {
        GGML_UNUSED(gg);
        const uint32_t oxj = j == 0 ? ox.x : j == 1 ? ox.y : j == 2 ? ox.z : ox.w;
        f0 = f1 = dsb * (0.5f + (float) (oxj >> 28)) * 0.5f;
        c0 = c1 = -128.0f*f0;
    }
};

struct qmma_dec_iq4_xs {
    static constexpr int WARPS = QMMA_BIG_WARPS;
    using blk = block_iq4_xs;
    static constexpr bool PER16 = false;
    struct smem { uint32_t ptab[256]; };
    static __device__ void init(smem & s) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            s.ptab[i] = (uint32_t) (uint8_t) (kvalues_iq4nl[i & 15] + 128) | ((uint32_t) (uint8_t) (kvalues_iq4nl[i >> 4] + 128) << 16);
        }
    }
    uint4 q[2]; uint64_t scq; float dsb;
    __device__ void load(const uint8_t * tb, const int nsb, const int sb, const int l, const int orow) {
        const int L = l & 63;
        q[0] = q[1] = make_uint4(0, 0, 0, 0);
        if (l < 64) {
            q[0] = ((const uint4 *) tb)[((int64_t) sb*64 + L)*2 + 0];
            q[1] = ((const uint4 *) tb)[((int64_t) sb*64 + L)*2 + 1];
        }
        const uint2 scv = ((const uint2 *) (tb + (int64_t) nsb*2048))[sb*16 + orow];
        scq = (uint64_t) scv.x | ((uint64_t) scv.y << 32);
        dsb = __half2float(__ushort_as_half((unsigned short) (scq & 0xFFFF)));
    }
    __device__ void load_gg(const int gg) { GGML_UNUSED(gg); }
    __device__ void decode(const smem & s, const int gg, const int j, const int l, uint32_t & a0, uint32_t & a1) const {
        GGML_UNUSED(l);
        const uint32_t v = j == 0 ? q[gg].x : j == 1 ? q[gg].y : j == 2 ? q[gg].z : q[gg].w;
        const uint32_t P = s.ptab[v & 0xFF] | (s.ptab[(v >> 8) & 0xFF] << 8);
        const uint32_t Y = s.ptab[(v >> 16) & 0xFF] | (s.ptab[v >> 24] << 8);
        a0 = (P & 0xFFFF) | (Y << 16);
        a1 = (P >> 16) | (Y & 0xFFFF0000u);
    }
    __device__ void coef(const int gg, const int j, float & f0, float & f1, float & c0, float & c1) const {
        const int ib = 4*gg + j;
        f0 = f1 = dsb * (float) ((int) ((scq >> (16 + 6*ib)) & 63) - 32);
        c0 = c1 = -128.0f*f0;
    }
};

struct qmma_dec_q4_K {
    static constexpr int WARPS = QMMA_BIG_WARPS;
    using blk = block_q4_K;
    static constexpr bool PER16 = false;
    struct smem { int dummy; };
    static __device__ void init(smem & s) { GGML_UNUSED(s); }
    uint4 q[2]; float2 dm; uint32_t scw[3];
    __device__ void load(const uint8_t * tb, const int nsb, const int sb, const int l, const int orow) {
        const int L = l & 63;
        q[0] = q[1] = make_uint4(0, 0, 0, 0);
        if (l < 64) {
            q[0] = ((const uint4 *) tb)[((int64_t) sb*64 + L)*2 + 0];
            q[1] = ((const uint4 *) tb)[((int64_t) sb*64 + L)*2 + 1];
        }
        const uint4 sc4 = ((const uint4 *) (tb + (int64_t) nsb*2048))[sb*16 + orow];
        dm = __half22float2(*(const half2 *) &sc4.x);
        scw[0] = sc4.y; scw[1] = sc4.z; scw[2] = sc4.w;
    }
    __device__ void load_gg(const int gg) { GGML_UNUSED(gg); }
    __device__ void decode(const smem & s, const int gg, const int j, const int l, uint32_t & a0, uint32_t & a1) const {
        GGML_UNUSED(s); GGML_UNUSED(l);
        const int ib = 4*gg + j, pw = 2*(j >> 1);
        const uint32_t qw[4] = {q[gg].x, q[gg].y, q[gg].z, q[gg].w};
        a0 = (ib & 1) ? (qw[pw]     >> 4) & 0x0F0F0F0F : qw[pw]     & 0x0F0F0F0F;
        a1 = (ib & 1) ? (qw[pw + 1] >> 4) & 0x0F0F0F0F : qw[pw + 1] & 0x0F0F0F0F;
    }
    __device__ void coef(const int gg, const int j, float & f0, float & f1, float & c0, float & c1) const {
        int sc, mn;
        qmma_q4k_scmin((const uint8_t *) scw, 4*gg + j, sc, mn);
        f0 = f1 = dm.x*(float) sc;
        c0 = c1 = -dm.y*(float) mn;
    }
};

struct qmma_dec_q5_K {
    static constexpr int WARPS = QMMA_BIG_WARPS;
    using blk = block_q5_K;
    static constexpr bool PER16 = false;
    struct smem { int dummy; };
    static __device__ void init(smem & s) { GGML_UNUSED(s); }
    uint4 q[2]; uint2 qh; float2 dm; uint32_t scw[3];
    __device__ void load(const uint8_t * tb, const int nsb, const int sb, const int l, const int orow) {
        const int L = l & 63;
        q[0] = q[1] = make_uint4(0, 0, 0, 0);
        qh = make_uint2(0, 0);
        if (l < 64) {
            q[0] = ((const uint4 *) tb)[((int64_t) sb*64 + L)*2 + 0];
            q[1] = ((const uint4 *) tb)[((int64_t) sb*64 + L)*2 + 1];
            qh   = ((const uint2 *) (tb + (int64_t) nsb*2048))[sb*64 + L];
        }
        const uint4 sc4 = ((const uint4 *) (tb + (int64_t) nsb*2560))[sb*16 + orow];
        dm = __half22float2(*(const half2 *) &sc4.x);
        scw[0] = sc4.y; scw[1] = sc4.z; scw[2] = sc4.w;
    }
    __device__ void load_gg(const int gg) { GGML_UNUSED(gg); }
    __device__ void decode(const smem & s, const int gg, const int j, const int l, uint32_t & a0, uint32_t & a1) const {
        GGML_UNUSED(s); GGML_UNUSED(l);
        const int ib = 4*gg + j, pw = 2*(j >> 1);
        const uint32_t qw[4] = {q[gg].x, q[gg].y, q[gg].z, q[gg].w};
        a0 = ((ib & 1) ? (qw[pw]     >> 4) & 0x0F0F0F0F : qw[pw]     & 0x0F0F0F0F) | (((qh.x >> ib) & 0x01010101u) << 4);
        a1 = ((ib & 1) ? (qw[pw + 1] >> 4) & 0x0F0F0F0F : qw[pw + 1] & 0x0F0F0F0F) | (((qh.y >> ib) & 0x01010101u) << 4);
    }
    __device__ void coef(const int gg, const int j, float & f0, float & f1, float & c0, float & c1) const {
        int sc, mn;
        qmma_q4k_scmin((const uint8_t *) scw, 4*gg + j, sc, mn);
        f0 = f1 = dm.x*(float) sc;
        c0 = c1 = -dm.y*(float) mn;
    }
};

struct qmma_dec_iq2_s {
    static constexpr int WARPS = QMMA_BIG_W_IQ2S;
    using blk = block_iq2_s;
    static constexpr bool PER16 = true;
    struct smem { uint32_t grid[2048]; };
    static __device__ void init(smem & s) {
        for (int i = threadIdx.x; i < 2048; i += blockDim.x) s.grid[i] = ((const uint32_t *) iq2s_grid)[i];
    }
    uint32_t iw[4], sw[4], qh; uint2 sc2; float dsb;
    __device__ void load(const uint8_t * tb, const int nsb, const int sb, const int l, const int orow) {
        const int L = l & 63, arow = L >> 2, cp = (L & 3) >> 1;
        uint4 idx = make_uint4(0, 0, 0, 0), sgn = idx;
        qh = 0;
        if (l < 64) {
            idx = ((const uint4 *) tb)[(sb*16 + arow)*2 + cp];
            sgn = ((const uint4 *) (tb + (int64_t) nsb*512))[(sb*16 + arow)*2 + cp];
            qh  = ((const uint32_t *) (tb + (int64_t) nsb*1024))[(sb*16 + arow)*2 + cp];
        }
        sc2 = ((const uint2 *) (tb + (int64_t) nsb*1152))[sb*16 + orow];
        dsb = __half2float(((const half *) (tb + (int64_t) nsb*1280))[sb*16 + orow]);
        iw[0] = idx.x; iw[1] = idx.y; iw[2] = idx.z; iw[3] = idx.w;
        sw[0] = sgn.x; sw[1] = sgn.y; sw[2] = sgn.z; sw[3] = sgn.w;
    }
    __device__ void load_gg(const int gg) { GGML_UNUSED(gg); }
    __device__ void decode(const smem & s, const int gg, const int j, const int l, uint32_t & a0, uint32_t & a1) const {
        const int ch = l & 1;
        const int ib = 4*gg + j;
        uint32_t av[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int bi = 2*ib + h;
            const uint32_t ib8 = (iw[bi >> 2] >> (8*(bi & 3))) & 0xFF;
            const uint32_t gv  = s.grid[2*(ib8 | (((qh >> (2*bi)) & 3) << 8)) + ch];
            const uint32_t nib = (sw[bi >> 2] >> (8*(bi & 3) + 4*ch)) & 0xF;
            const uint32_t m01 = (nib * 0x00204081u) & 0x01010101u;
            av[h] = ((gv + 0x40404040u) ^ (m01 * 0x7Fu)) + m01;
        }
        a0 = av[0]; a1 = av[1];
    }
    __device__ void coef(const int gg, const int j, float & f0, float & f1, float & c0, float & c1) const {
        const int ib = 4*gg + j;
        const uint32_t scb = ((ib < 4 ? sc2.x : sc2.y) >> (8*(ib & 3))) & 0xFF;
        f0 = dsb*(0.5f + (float) (scb & 0xF))*0.25f;
        f1 = dsb*(0.5f + (float) (scb >> 4))*0.25f;
        c0 = -64.0f*f0; c1 = -64.0f*f1;
    }
};

struct qmma_dec_q2_K {
    static constexpr int WARPS = QMMA_BIG_WARPS;
    using blk = block_q2_K;
    static constexpr bool PER16 = true;
    struct smem { int dummy; };
    static __device__ void init(smem & s) { GGML_UNUSED(s); }
    uint32_t qw[4], scw[4]; float2 dm;
    __device__ void load(const uint8_t * tb, const int nsb, const int sb, const int l, const int orow) {
        const int L = l & 63;
        uint4 qw4 = make_uint4(0, 0, 0, 0);
        if (l < 64) {
            qw4 = ((const uint4 *) tb)[sb*64 + L];
        }
        const uint4 sc4 = ((const uint4 *) (tb + (int64_t) nsb*1024))[sb*16 + orow];
        dm = __half22float2(*(const half2 *) (tb + (int64_t) nsb*1280 + ((int64_t) sb*16 + orow)*4));
        qw[0] = qw4.x; qw[1] = qw4.y; qw[2] = qw4.z; qw[3] = qw4.w;
        scw[0] = sc4.x; scw[1] = sc4.y; scw[2] = sc4.z; scw[3] = sc4.w;
    }
    __device__ void load_gg(const int gg) { GGML_UNUSED(gg); }
    __device__ void decode(const smem & s, const int gg, const int j, const int l, uint32_t & a0, uint32_t & a1) const {
        GGML_UNUSED(s); GGML_UNUSED(l);
        a0 = (qw[2*gg]     >> (2*j)) & 0x03030303u;
        a1 = (qw[2*gg + 1] >> (2*j)) & 0x03030303u;
    }
    __device__ void coef(const int gg, const int j, float & f0, float & f1, float & c0, float & c1) const {
        const int is = 8*gg + 2*j;
        const uint32_t s0 = (scw[is >> 2] >> (8*(is & 3))) & 0xFF, s1 = (scw[(is + 1) >> 2] >> (8*((is + 1) & 3))) & 0xFF;
        f0 = dm.x*(float) (s0 & 0xF); f1 = dm.x*(float) (s1 & 0xF);
        c0 = -dm.y*(float) (s0 >> 4); c1 = -dm.y*(float) (s1 >> 4);
    }
};

struct qmma_dec_q6_K {
    static constexpr int WARPS = QMMA_BIG_WARPS;
    using blk = block_q6_K;
    static constexpr bool PER16 = true;
    struct smem { int dummy; };
    static __device__ void init(smem & s) { GGML_UNUSED(s); }
    uint4 ql[2]; uint2 qh[2]; uint4 sc4; float dsb;
    __device__ void load(const uint8_t * tb, const int nsb, const int sb, const int l, const int orow) {
        const int L = l & 63;
        ql[0] = ql[1] = make_uint4(0, 0, 0, 0); qh[0] = qh[1] = make_uint2(0, 0);
        if (l < 64) {
#pragma unroll
            for (int gg = 0; gg < 2; ++gg) {
                ql[gg] = ((const uint4 *) tb)[(sb*64 + L)*2 + gg];
                qh[gg] = ((const uint2 *) (tb + (int64_t) nsb*2048))[(sb*64 + L)*2 + gg];
            }
        }
        sc4 = ((const uint4 *) (tb + (int64_t) nsb*3072))[sb*16 + orow];
        dsb = __half2float(((const half *) (tb + (int64_t) nsb*3328))[sb*16 + orow]);
    }
    __device__ void load_gg(const int gg) { GGML_UNUSED(gg); }
    __device__ void decode(const smem & s, const int gg, const int j, const int l, uint32_t & a0, uint32_t & a1) const {
        GGML_UNUSED(s); GGML_UNUSED(l);
        const uint32_t qlw[4] = {ql[gg].x, ql[gg].y, ql[gg].z, ql[gg].w};
        const int part = j & 1, nsh = 4*(j >> 1);
        a0 = ((qlw[part*2]     >> nsh) & 0x0F0F0F0Fu) | (((qh[gg].x >> (2*j)) & 0x03030303u) << 4);
        a1 = ((qlw[part*2 + 1] >> nsh) & 0x0F0F0F0Fu) | (((qh[gg].y >> (2*j)) & 0x03030303u) << 4);
    }
    __device__ void coef(const int gg, const int j, float & f0, float & f1, float & c0, float & c1) const {
        const int8_t * scb = (const int8_t *) &sc4;
        f0 = dsb*(float) scb[8*gg + 2*j]; f1 = dsb*(float) scb[8*gg + 2*j + 1];
        c0 = -32.0f*f0; c1 = -32.0f*f1;
    }
};

struct qmma_dec_iq2_xxs {
    static constexpr int WARPS = QMMA_BIG_WARPS;
    using blk = block_iq2_xxs;
    static constexpr bool PER16 = false;
    struct smem { uint32_t grid[512]; uint32_t stab[256]; };
    static __device__ void init(smem & s) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) s.grid[i] = ((const uint32_t *) iq2xxs_grid)[i];
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            const uint32_t g7 = i >> 1, s8 = g7 | ((__popc(g7) & 1) << 7), nb4 = (s8 >> (4*(i & 1))) & 0xF;
            s.stab[i] = (nb4 * 0x00204081u) & 0x01010101u;
        }
    }
    uint32_t ix[8], ax[8], ox[8]; float dsb;
    __device__ void load(const uint8_t * tb, const int nsb, const int sb, const int l, const int orow) {
        const int L = l & 63, arow = L >> 2;
        const uint4 * ixp = (const uint4 *) tb;
        const uint4 * axp = (const uint4 *) (tb + (int64_t) nsb*512);
        uint4 ix0 = make_uint4(0, 0, 0, 0), ix1 = ix0, ax0 = ix0, ax1 = ix0;
        if (l < 64) {
            ix0 = ixp[(sb*16 + arow)*2]; ix1 = ixp[(sb*16 + arow)*2 + 1];
            ax0 = axp[(sb*16 + arow)*2]; ax1 = axp[(sb*16 + arow)*2 + 1];
        }
        const uint4 ox0 = axp[(sb*16 + orow)*2], ox1 = axp[(sb*16 + orow)*2 + 1];
        dsb = __half2float(((const half *) (tb + (int64_t) nsb*1024))[sb*16 + orow]);
        ix[0] = ix0.x; ix[1] = ix0.y; ix[2] = ix0.z; ix[3] = ix0.w; ix[4] = ix1.x; ix[5] = ix1.y; ix[6] = ix1.z; ix[7] = ix1.w;
        ax[0] = ax0.x; ax[1] = ax0.y; ax[2] = ax0.z; ax[3] = ax0.w; ax[4] = ax1.x; ax[5] = ax1.y; ax[6] = ax1.z; ax[7] = ax1.w;
        ox[0] = ox0.x; ox[1] = ox0.y; ox[2] = ox0.z; ox[3] = ox0.w; ox[4] = ox1.x; ox[5] = ox1.y; ox[6] = ox1.z; ox[7] = ox1.w;
    }
    __device__ void load_gg(const int gg) { GGML_UNUSED(gg); }
    __device__ void decode(const smem & s, const int gg, const int j, const int l, uint32_t & a0, uint32_t & a1) const {
        const int c = l & 3, cp = c >> 1, ch = c & 1;
        const int ib = 4*gg + j;
        uint32_t av[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int li = 2*h + cp;
            const uint32_t gv  = s.grid[2*((ix[ib] >> (8*li)) & 0xFF) + ch];
            const uint32_t m01 = s.stab[(((ax[ib] >> (7*li)) & 0x7F) << 1) | ch];
            av[h] = (gv ^ (0x80808080u - m01)) + m01;
        }
        a0 = av[0]; a1 = av[1];
    }
    __device__ void coef(const int gg, const int j, float & f0, float & f1, float & c0, float & c1) const {
        const int ib = 4*gg + j;
        f0 = f1 = dsb * (0.5f + (float) (ox[ib] >> 28)) * 0.25f;
        c0 = c1 = -128.0f*f0;
    }
};

struct qmma_dec_iq2_xs {
    static constexpr int WARPS = QMMA_BIG_W_IQ2XS;
    using blk = block_iq2_xs;
    static constexpr bool PER16 = true;
    struct smem { uint32_t grid[1024]; uint32_t stab[256]; };
    static __device__ void init(smem & s) {
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) s.grid[i] = ((const uint32_t *) iq2xs_grid)[i];
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            const uint32_t g7 = i >> 1, s8 = g7 | ((__popc(g7) & 1) << 7), nb4 = (s8 >> (4*(i & 1))) & 0xF;
            s.stab[i] = (nb4 * 0x00204081u) & 0x01010101u;
        }
    }
    uint32_t qw[8]; uint2 sc2; float dsb;
    __device__ void load(const uint8_t * tb, const int nsb, const int sb, const int l, const int orow) {
        const int L = l & 63, arow = L >> 2, cp = (L & 3) >> 1;
        uint4 q0 = make_uint4(0, 0, 0, 0), q1 = q0;
        if (l < 64) {
            q0 = ((const uint4 *) tb)[((sb*16 + arow)*2 + cp)*2];
            q1 = ((const uint4 *) tb)[((sb*16 + arow)*2 + cp)*2 + 1];
        }
        sc2 = ((const uint2 *) (tb + (int64_t) nsb*1024))[sb*16 + orow];
        dsb = __half2float(((const half *) (tb + (int64_t) nsb*1152))[sb*16 + orow]);
        qw[0] = q0.x; qw[1] = q0.y; qw[2] = q0.z; qw[3] = q0.w; qw[4] = q1.x; qw[5] = q1.y; qw[6] = q1.z; qw[7] = q1.w;
    }
    __device__ void load_gg(const int gg) { GGML_UNUSED(gg); }
    __device__ void decode(const smem & s, const int gg, const int j, const int l, uint32_t & a0, uint32_t & a1) const {
        const int ch = l & 1;
        const int ib = 4*gg + j;
        uint32_t av[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t v   = (qw[ib] >> (16*h)) & 0xFFFF;
            const uint32_t gv  = s.grid[2*(v & 511) + ch];
            const uint32_t m01 = s.stab[((v >> 9) << 1) | ch];
            av[h] = (gv ^ (0x80808080u - m01)) + m01;
        }
        a0 = av[0]; a1 = av[1];
    }
    __device__ void coef(const int gg, const int j, float & f0, float & f1, float & c0, float & c1) const {
        const int ib = 4*gg + j;
        const uint32_t scb = ((ib < 4 ? sc2.x : sc2.y) >> (8*(ib & 3))) & 0xFF;
        f0 = dsb*(0.5f + (float) (scb & 0xF))*0.25f;
        f1 = dsb*(0.5f + (float) (scb >> 4))*0.25f;
        c0 = -128.0f*f0; c1 = -128.0f*f1;
    }
};

// MMA with an initial accumulator: C = -128*S folds the row-sum correction into the tensor core
__device__ __forceinline__ int qmma_block_c(const uint32_t a0v, const uint32_t a1v, const uint32_t b0v, const uint32_t b1v, const int cinit) {
    int a0[2] = {(int) a0v, 0}, a1[2] = {(int) a1v, 0};
    int b0[1] = {(int) b0v}, b1[1] = {(int) b1v};
    int c[4] = {cinit, 0, 0, 0}, d[4] = {0, 0, 0, 0};
    __musa_imma_m16n8k16_mma(d, a0, b0, c, 0, 0, 0, false, 0, 1);
    c[0] = d[0];
    __musa_imma_m16n8k16_mma(d, a1, b1, c, 0, 0, 0, false, 0, 1);
    return d[0];
}
__device__ __forceinline__ int qmma_one_c(const uint32_t av, const uint32_t bv, const int cinit) {
    int a[2] = {(int) av, 0}, b[1] = {(int) bv}, c[4] = {cinit, 0, 0, 0}, d[4] = {0, 0, 0, 0};
    __musa_imma_m16n8k16_mma(d, a, b, c, 0, 0, 0, false, 0, 1);
    return d[0];
}

// NG token groups of 8 per launch. The activation buffers are padded to whole groups (garbage tokens are
// computed but never stored), so the token loop has no bounds checks.
template <class Dec, int NG>
__launch_bounds__(Dec::WARPS*128, 1)
__global__ void qmma_big(const uint8_t * __restrict__ w, const uint4 * __restrict__ ya, const float4 * __restrict__ da,
                         const float4 * __restrict__ dl, float * __restrict__ dst, const int64_t stride_dst_tok,
                         const int R, const int nsb, const int N, const int ones) {
    __shared__ typename Dec::smem sm;
    Dec::init(sm);
    __syncthreads();
    const int l = threadIdx.x & 127, wp = threadIdx.x >> 7;
    const int tile = blockIdx.x*Dec::WARPS + wp;
    if (tile*16 >= R) {
        return;
    }
    const int ngrp = nsb*2;
    const int bt = l >> 2, orow = l >> 3, ocol = l & 7;
    const uint8_t * tb = w + (int64_t) tile*16*nsb*(int64_t) sizeof(typename Dec::blk);
    // per token group G: activation fragments at yb + G*ysG + g*8, scales at db + G*dsG + g*2, dl at lb + G*lsG + g
    const uint4  * yb = ya + (int64_t) bt*ngrp*8 + (l & 3)*2;
    const float4 * db = da + (int64_t) ocol*ngrp*2;
    const float4 * lb = dl + (int64_t) ocol*ngrp;
    const int ysG = 8*ngrp*8, dsG = 8*ngrp*2, lsG = 8*ngrp;
    const int ngr = (N + 7)/8;
    float acc[NG];
#pragma unroll
    for (int G = 0; G < NG; ++G) {
        acc[G] = 0.0f;
    }
    for (int sb = 0; sb < nsb; ++sb) {
        Dec dec;
        dec.load(tb, nsb, sb, l, orow);
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) {
            dec.load_gg(gg);
            const int g = 2*sb + gg;
            uint32_t a0[4], a1[4];
            int C0[4], C1[4];
            float f0[4], f1[4], c0[4], c1[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                dec.decode(sm, gg, j, l, a0[j], a1[j]);
                dec.coef(gg, j, f0[j], f1[j], c0[j], c1[j]);
                if constexpr (Dec::PER16) {
                    C0[j] = -128*qmma_one(a0[j], ones);
                    C1[j] = -128*qmma_one(a1[j], ones);
                } else {
                    C0[j] = -128*qmma_block(a0[j], a1[j], ones, ones);
                    C1[j] = 0;
                }
            }
#pragma unroll
            for (int G = 0; G < NG; ++G) {
                if (G >= ngr) {
                    break;   // uniform; also stops the scheduler from interleaving every group (register blow-up)
                }
                uint4 ylo = make_uint4(0, 0, 0, 0), yhi = ylo;
                if (l < 32) {
                    const uint4 * yp = yb + G*ysG + g*8;
                    ylo = yp[0];
                    yhi = yp[1];
                }
                const uint32_t yv[8] = {ylo.x, ylo.y, ylo.z, ylo.w, yhi.x, yhi.y, yhi.z, yhi.w};
                const float4 ds01 = db[G*dsG + g*2], ds23 = db[G*dsG + g*2 + 1];
                const float dsx[4] = {ds01.x, ds01.z, ds23.x, ds23.z}, dsy[4] = {ds01.y, ds01.w, ds23.y, ds23.w};
                if constexpr (Dec::PER16) {
                    const float4 lo4 = lb[G*lsG + g];
                    const float dlo[4] = {lo4.x, lo4.y, lo4.z, lo4.w};
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int D0 = qmma_one_c(a0[j], yv[2*j], C0[j]), D1 = qmma_one_c(a1[j], yv[2*j + 1], C1[j]);
                        acc[G] = fmaf(dsx[j], fmaf(f0[j], (float) D0, f1[j]*(float) D1), acc[G]);
                        acc[G] = fmaf(c0[j] - c1[j], dlo[j], fmaf(c1[j], dsy[j], acc[G]));
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int D = qmma_block_c(a0[j], a1[j], yv[2*j], yv[2*j + 1], C0[j]);
                        acc[G] = fmaf(dsx[j]*f0[j], (float) D, fmaf(c0[j], dsy[j], acc[G]));
                    }
                }
                // keep the compiler from hoisting every group's loads up front (registers explode to 254 + spills)
                if (G % QMMA_BIG_PIPE == QMMA_BIG_PIPE - 1) {
                    asm volatile("" ::: "memory");
                }
            }
        }
    }
    const int r = tile*16 + orow;
#pragma unroll
    for (int G = 0; G < NG; ++G) {
        const int t = 8*G + ocol;
        if (t < N && r < R) {
            dst[t*stride_dst_tok + r] = acc[G];
        }
    }
}


#define QMMA_KS_DISPATCH(K, BASE, ARGS) do {                                                                      \
        if (ks == 1) {                                                                                       \
            K<1><<<(ntile + (BASE) - 1)/(BASE), (BASE)*128, 0, st>>> ARGS;                                   \
        } else if (ks == 2) {                                                                                \
            K<2><<<(ntile + 1)/2, 512, 0, st>>> ARGS;                                                        \
        } else {                                                                                             \
            K<4><<<ntile, 512, 0, st>>> ARGS;                                                                \
        }                                                                                                    \
    } while (0)

// A view of the first rows of an already-tiled weight (e.g. a truncated MTP draft LM head) reads the
// same tiles: tile t covers rows 16t..16t+15 in place, so a row prefix that is a multiple of 16 is valid.
bool qmma_prefix_view_ok(const ggml_tensor * t) {
    const ggml_tensor * s = t->view_src;
    if (t->view_offs != 0 || s->type != t->type || s->ne[0] != t->ne[0] || s->nb[1] != t->nb[1] || t->data != s->data) {
        return false;
    }
    std::lock_guard<std::mutex> lk(g_tiled_lock);
    return g_tiled.count(s->data) != 0;
}

bool qmma_shape_ok(const ggml_tensor * t) {
    int64_t kmul;
    switch (t->type) {
        case GGML_TYPE_Q4_0:   kmul = 256; break;
        case GGML_TYPE_Q4_1:   kmul = 256; break;
        case GGML_TYPE_Q5_K:   kmul = 256; break;
        case GGML_TYPE_Q8_0:   kmul = 256; break;
        case GGML_TYPE_IQ4_XS: kmul = 256; break;
        case GGML_TYPE_IQ3_S:  kmul = 256; break;
        case GGML_TYPE_IQ3_XXS:kmul = 256; break;
        case GGML_TYPE_Q4_K:   kmul = 256; break;
        case GGML_TYPE_IQ2_S:  kmul = 256; break;
        case GGML_TYPE_Q2_K:   kmul = 256; break;
        case GGML_TYPE_Q6_K:   kmul = 256; break;
        case GGML_TYPE_IQ2_XXS:kmul = 256; break;
        case GGML_TYPE_IQ2_XS: kmul = 256; break;
        default: return false;
    }
    return qmma_type_enabled(t->type) && ggml_is_contiguous(t) && t->ne[2] == 1 && t->ne[3] == 1 &&
           t->ne[0] % kmul == 0 && t->ne[1] % 16 == 0 && t->buffer != nullptr &&
           ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
           strstr(t->name, "token_embd") == nullptr && (t->view_src == nullptr || qmma_prefix_view_ok(t));
}

void qmma_repack(cudaStream_t st, ggml_type type, uint8_t * base, const int64_t K, const int ntile) {
    // one small persistent scratch (not the pool: with a ~15 GB model the pool cannot grow mid-run)
    static uint8_t * s_scratch = nullptr;
    static size_t    s_scratch_size = 0;
    const size_t tile_bytes = (size_t) 16*ggml_row_size(type, K);
    const size_t want = std::max<size_t>(tile_bytes, 2u << 20);
    if (s_scratch_size < want) {
        if (s_scratch) {
            CUDA_CHECK(cudaStreamSynchronize(st));
            CUDA_CHECK(cudaFree(s_scratch));
        }
        CUDA_CHECK(cudaMalloc((void **) &s_scratch, want));
        s_scratch_size = want;
    }
    const int chunk = (int) std::max<size_t>(1, std::min<size_t>(ntile, s_scratch_size/tile_bytes));
    for (int t0 = 0; t0 < ntile; t0 += chunk) {
        const int nt = std::min(chunk, ntile - t0);
        uint8_t * part = base + (size_t) t0*tile_bytes;
        CUDA_CHECK(cudaMemcpyAsync(s_scratch, part, (size_t) nt*tile_bytes, cudaMemcpyDeviceToDevice, st));
        switch (type) {
            case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: {
                const int nsb = (int) (K/256);
                const int64_t nq = (int64_t) nt*nsb*64, nd = (int64_t) nt*nsb*16;
                if (type == GGML_TYPE_Q4_0) {
                    qmma_repack_q4_x_q<block_q4_0><<<(nq + 255)/256, 256, 0, st>>>((const block_q4_0 *) s_scratch, part, nt, nsb);
                    qmma_repack_q4_0_d<<<(nd + 255)/256, 256, 0, st>>>((const block_q4_0 *) s_scratch, part, nt, nsb);
                } else {
                    qmma_repack_q4_x_q<block_q4_1><<<(nq + 255)/256, 256, 0, st>>>((const block_q4_1 *) s_scratch, part, nt, nsb);
                    qmma_repack_q4_1_d<<<(nd + 255)/256, 256, 0, st>>>((const block_q4_1 *) s_scratch, part, nt, nsb);
                }
            } break;
            case GGML_TYPE_Q5_K: {
                const int nsb = (int) (K/256);
                const int64_t nq = (int64_t) nt*nsb*64, nd = (int64_t) nt*nsb*16;
                qmma_repack_q5_K_q<<<(nq + 255)/256, 256, 0, st>>>((const block_q5_K *) s_scratch, part, nt, nsb);
                qmma_repack_q5_K_d<<<(nd + 255)/256, 256, 0, st>>>((const block_q5_K *) s_scratch, part, nt, nsb);
            } break;
            case GGML_TYPE_IQ4_XS: {
                const int nsb = (int) (K/256);
                const int64_t nq = (int64_t) nt*nsb*64, nd = (int64_t) nt*nsb*16;
                qmma_repack_iq4_xs_q<<<(nq + 255)/256, 256, 0, st>>>((const block_iq4_xs *) s_scratch, part, nt, nsb);
                qmma_repack_iq4_xs_d<<<(nd + 255)/256, 256, 0, st>>>((const block_iq4_xs *) s_scratch, part, nt, nsb);
            } break;
            case GGML_TYPE_IQ3_S: {
                const int nsb = (int) (K/256);
                const int64_t nq = (int64_t) nt*nsb*64, nd = (int64_t) nt*nsb*16;
                qmma_repack_iq3_s_q<<<(nq + 255)/256, 256, 0, st>>>((const block_iq3_s *) s_scratch, part, nt, nsb);
                qmma_repack_iq3_s_d<<<(nd + 255)/256, 256, 0, st>>>((const block_iq3_s *) s_scratch, part, nt, nsb);
            } break;
            case GGML_TYPE_IQ3_XXS: {
                const int nsb = (int) (K/256);
                const int64_t nq = (int64_t) nt*nsb*64, nd = (int64_t) nt*nsb*16;
                qmma_repack_iq3_xxs_q<<<(nq + 255)/256, 256, 0, st>>>((const block_iq3_xxs *) s_scratch, part, nt, nsb);
                qmma_repack_iq3_xxs_d<<<(nd + 255)/256, 256, 0, st>>>((const block_iq3_xxs *) s_scratch, part, nt, nsb);
            } break;
            case GGML_TYPE_Q4_K: {
                const int nsb = (int) (K/256);
                const int64_t nq = (int64_t) nt*nsb*64, nd = (int64_t) nt*nsb*16;
                qmma_repack_q4_K_q<<<(nq + 255)/256, 256, 0, st>>>((const block_q4_K *) s_scratch, part, nt, nsb);
                qmma_repack_q4_K_d<<<(nd + 255)/256, 256, 0, st>>>((const block_q4_K *) s_scratch, part, nt, nsb);
            } break;
            case GGML_TYPE_IQ2_S: {
                const int nsb = (int) (K/256);
                const int64_t nd = (int64_t) nt*nsb*16;
                qmma_repack_iq2_s<<<(nd + 255)/256, 256, 0, st>>>((const block_iq2_s *) s_scratch, part, nt, nsb);
            } break;
            case GGML_TYPE_Q2_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: {
                const int nsb = (int) (K/256);
                const int64_t nl = (int64_t) nt*nsb*64, nr = (int64_t) nt*nsb*16;
                if (type == GGML_TYPE_Q2_K) {
                    qmma_repack_q2_K<<<(nl + 255)/256, 256, 0, st>>>((const block_q2_K *) s_scratch, part, nt, nsb);
                } else if (type == GGML_TYPE_Q6_K) {
                    qmma_repack_q6_K<<<(nl + 255)/256, 256, 0, st>>>((const block_q6_K *) s_scratch, part, nt, nsb);
                } else if (type == GGML_TYPE_IQ2_XXS) {
                    qmma_repack_iq2_xxs<<<(nr + 255)/256, 256, 0, st>>>((const block_iq2_xxs *) s_scratch, part, nt, nsb);
                } else {
                    qmma_repack_iq2_xs<<<(nr + 255)/256, 256, 0, st>>>((const block_iq2_xs *) s_scratch, part, nt, nsb);
                }
            } break;
            case GGML_TYPE_Q8_0: {
                const int nsb = (int) (K/256);
                const int64_t nq = (int64_t) nt*nsb*64, nd = (int64_t) nt*nsb*16;
                qmma_repack_q8_0_q<<<(nq + 255)/256, 256, 0, st>>>((const block_q8_0 *) s_scratch, part, nt, nsb);
                qmma_repack_q8_0_d<<<(nd + 255)/256, 256, 0, st>>>((const block_q8_0 *) s_scratch, part, nt, nsb);
            } break;
            default:
                GGML_ABORT("qmma_repack: unsupported type");
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

// True if this weight tensor is (or will be) handled by the tensor-core path. Fusion code uses this
// to avoid sending such tensors to kernels that expect the original layout.
uint64_t ggml_cuda_qmma_epoch = 0;

bool ggml_cuda_q4_0_mma_eligible(const ggml_tensor * src0) {
    return qmma_enabled() && qmma_shape_ok(src0);
}

bool ggml_cuda_q4_0_mma_is_tiled(const ggml_tensor * t) {
    if (t == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lk(g_tiled_lock);
    return g_tiled.count(t->data) != 0;
}

// kept for the upload hook in ggml-cuda.cu: tiling now happens on first use, on the GPU
bool ggml_cuda_q4_0_upload_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(t); GGML_UNUSED(data); GGML_UNUSED(offset); GGML_UNUSED(size);
    return false;
}

bool ggml_cuda_mul_mat_q4_0_mma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (!ggml_cuda_q4_0_mma_eligible(src0)) {
        return false;
    }
    const int64_t K = src0->ne[0], R = src0->ne[1], N = src1->ne[1];
    const bool ok_shapes = src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src1->ne[0] == K &&
                           src1->ne[2] == 1 && src1->ne[3] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1 &&
                           src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
                           src1->nb[1] % sizeof(float) == 0 && dst->nb[1] % sizeof(float) == 0 &&
                           ((uintptr_t) src1->data % 16) == 0 && (src1->nb[1] % 16) == 0;
    bool tiled;
    {
        std::lock_guard<std::mutex> lk(g_tiled_lock);
        tiled = g_tiled.count(src0->data) != 0;
    }
    if (!ok_shapes) {
        // the weights may already be tiled: nothing else can read them
        GGML_ASSERT(!tiled && "MUSA tensor-core path: unsupported src1/dst layout for a tiled weight");
        return false;
    }
    cudaStream_t st = ctx.stream();
    const int nb = (int) (K/32), ngrp = nb/4, ntile = (int) (R/16), nsb = (int) (K/256);

    if (!tiled) {
        qmma_repack(st, src0->type, (uint8_t *) src0->data, K, ntile);
        std::lock_guard<std::mutex> lk(g_tiled_lock);
        g_tiled.insert(src0->data);
    }

    // activations
    static const bool big_on = getenv("GGML_MUSA_QMMA_BIG") == nullptr || atoi(getenv("GGML_MUSA_QMMA_BIG")) != 0;
    const bool use_big = big_on && N > QMMA_MAX_TOK;
    const int64_t Npad = use_big ? (N + QMMA_BIG_NG*8 - 1)/(QMMA_BIG_NG*8)*(QMMA_BIG_NG*8) : N;   // big kernel reads whole groups
    // Decode-sized batches: one-entry cache of the quantized activations. Consecutive matmuls that read the same
    // input (q/k/v, ffn gate/up, DeltaNet qkv/z) quantize it once. Valid only within one graph compute
    // (ggml_cuda_qmma_epoch is bumped by ggml_backend_cuda_graph_compute); the graph allocator never overwrites a
    // tensor that still has pending readers, so the same tensor + data pointer means the same values.
    static const bool qcache_on = getenv("GGML_MUSA_QCACHE") == nullptr || atoi(getenv("GGML_MUSA_QCACHE")) != 0;
    constexpr int64_t QC_TOK = 8, QC_K = 32768;
    static uint32_t * qc_ya = nullptr;
    static float2   * qc_da = nullptr;
    static float    * qc_dl = nullptr;
    static const ggml_tensor * qc_src = nullptr;
    static const void * qc_data = nullptr;
    static int64_t qc_K = 0, qc_N = 0, qc_nb1 = 0;
    static uint64_t qc_epoch = ~0ull;
    const bool use_qc = qcache_on && !use_big && N <= QC_TOK && K <= QC_K;
    ggml_cuda_pool_alloc<uint32_t> ya_pool;
    ggml_cuda_pool_alloc<float2>   da_pool;
    ggml_cuda_pool_alloc<float>    dl_pool;
    uint32_t * ya_p; float2 * da_p; float * dl_p;
    bool need_quant = true;
    if (use_qc) {
        if (qc_ya == nullptr) {
            CUDA_CHECK(cudaMalloc(&qc_ya, (size_t) QC_TOK*(QC_K/128)*32*sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&qc_da, (size_t) QC_TOK*(QC_K/128)*4*sizeof(float2)));
            CUDA_CHECK(cudaMalloc(&qc_dl, (size_t) QC_TOK*(QC_K/128)*4*sizeof(float)));
        }
        ya_p = qc_ya; da_p = qc_da; dl_p = qc_dl;
        need_quant = !(qc_src == src1 && qc_data == src1->data && qc_K == K && qc_N == N &&
                       qc_nb1 == (int64_t) src1->nb[1] && qc_epoch == ggml_cuda_qmma_epoch);
        qc_src = src1; qc_data = src1->data; qc_K = K; qc_N = N; qc_nb1 = (int64_t) src1->nb[1]; qc_epoch = ggml_cuda_qmma_epoch;
    } else {
        ya_p = ya_pool.alloc(ctx.pool(), (size_t) Npad*ngrp*32);
        da_p = da_pool.alloc(ctx.pool(), (size_t) Npad*ngrp*4);
        dl_p = dl_pool.alloc(ctx.pool(), (size_t) Npad*ngrp*4);
        qc_src = nullptr;   // the cached buffer is untouched, but keep the rule simple: any other matmul invalidates
    }
    if (need_quant) {
        const dim3 grid((nb + QMMA_QBLK - 1)/QMMA_QBLK, (unsigned) N);
        qmma_quantize<<<grid, QMMA_QBLK*8, 0, st>>>((const float *) src1->data, src1->nb[1]/sizeof(float), ya_p, da_p, dl_p, nb, (int) N);
    }
    const int64_t sdst = dst->nb[1]/sizeof(float);
    // debug/profiling: GGML_MUSA_QMMA_SKIP=1 skips the matmul kernels (output is garbage) to measure the rest of the graph
    static const bool skip = getenv("GGML_MUSA_QMMA_SKIP") != nullptr && atoi(getenv("GGML_MUSA_QMMA_SKIP")) != 0;
    if (skip) {
        return true;
    }
    if (use_big) {
        const int nblk = (ntile + QMMA_BIG_WARPS - 1)/QMMA_BIG_WARPS;
        static const int big_dbg = getenv("GGML_MUSA_QMMA_BIG_DBG") ? atoi(getenv("GGML_MUSA_QMMA_BIG_DBG")) : 0;
        const int tpl  = QMMA_BIG_NG*8;
        for (int64_t t0 = 0; t0 < N; t0 += tpl) {
            const int nt = (int) std::min<int64_t>(tpl, N - t0);
            const uint4  * yat = (const uint4 *)  (ya_p + (size_t) t0*ngrp*32);
            const float4 * dat = (const float4 *) (da_p + (size_t) t0*ngrp*4);
            const float4 * dlt = (const float4 *) (dl_p + (size_t) t0*ngrp*4);
            float * dt = (float *) dst->data + t0*sdst;
        if (big_dbg & 256) {
            static bool once = false;
            if (!once) {
                once = true;
                musaFuncAttributes fa;
#define QMMA_ATTR(DEC) musaFuncGetAttributes(&fa, (const void *) qmma_big<DEC, 8>); fprintf(stderr, "%-18s regs %d local %zu shared %zu\n", #DEC, fa.numRegs, fa.localSizeBytes, fa.sharedSizeBytes);
                QMMA_ATTR(qmma_dec_iq3_s) QMMA_ATTR(qmma_dec_iq3_xxs) QMMA_ATTR(qmma_dec_iq4_xs) QMMA_ATTR(qmma_dec_q4_K)
                QMMA_ATTR(qmma_dec_iq2_s) QMMA_ATTR(qmma_dec_q2_K) QMMA_ATTR(qmma_dec_q6_K) QMMA_ATTR(qmma_dec_iq2_xxs) QMMA_ATTR(qmma_dec_iq2_xs)
#undef QMMA_ATTR
            }
        }
#define QMMA_BIG_LAUNCH_NG(DEC, NG) qmma_big<DEC, NG><<<(ntile + DEC::WARPS - 1)/DEC::WARPS, DEC::WARPS*128, 0, st>>>((const uint8_t *) src0->data, yat, dat, dlt, dt, sdst, (int) R, nsb, nt, 0x01010101)
#define QMMA_BIG_LAUNCH(DEC) do { const int ngr = (nt + 7)/8; \
                if (ngr > 4) { QMMA_BIG_LAUNCH_NG(DEC, 8); } else if (ngr > 2) { QMMA_BIG_LAUNCH_NG(DEC, 4); } \
                else if (ngr > 1) { QMMA_BIG_LAUNCH_NG(DEC, 2); } else { QMMA_BIG_LAUNCH_NG(DEC, 1); } } while (0)
            switch (src0->type) {
                case GGML_TYPE_IQ3_S:   QMMA_BIG_LAUNCH(qmma_dec_iq3_s);   break;
                case GGML_TYPE_IQ3_XXS: QMMA_BIG_LAUNCH(qmma_dec_iq3_xxs); break;
                case GGML_TYPE_IQ4_XS:  QMMA_BIG_LAUNCH(qmma_dec_iq4_xs);  break;
                case GGML_TYPE_Q4_K:    QMMA_BIG_LAUNCH(qmma_dec_q4_K);    break;
                case GGML_TYPE_IQ2_S:   QMMA_BIG_LAUNCH(qmma_dec_iq2_s);   break;
                case GGML_TYPE_Q2_K:    QMMA_BIG_LAUNCH(qmma_dec_q2_K);    break;
                case GGML_TYPE_Q6_K:    QMMA_BIG_LAUNCH(qmma_dec_q6_K);    break;
                case GGML_TYPE_IQ2_XXS: QMMA_BIG_LAUNCH(qmma_dec_iq2_xxs); break;
                case GGML_TYPE_IQ2_XS:  QMMA_BIG_LAUNCH(qmma_dec_iq2_xs);  break;
                case GGML_TYPE_Q8_0:    QMMA_BIG_LAUNCH(qmma_dec_q8_0);    break;
                case GGML_TYPE_Q4_0:    QMMA_BIG_LAUNCH(qmma_dec_q4_0);    break;
                case GGML_TYPE_Q4_1:    QMMA_BIG_LAUNCH(qmma_dec_q4_1);    break;
                case GGML_TYPE_Q5_K:    QMMA_BIG_LAUNCH(qmma_dec_q5_K);    break;
                default: GGML_ABORT("qmma_big: unhandled type");
            }
#undef QMMA_BIG_LAUNCH
#undef QMMA_BIG_LAUNCH_NG
        }
        return true;
    }
    {
        static bool attr_once = false;
        if (!attr_once && getenv("GGML_MUSA_QMMA_ATTR")) {
            attr_once = true;
            musaFuncAttributes fa;
#define QMMA_SATTR(K, T) musaFuncGetAttributes(&fa, (const void *) K); \
            fprintf(stderr, "%-28s regs %3d local %4zu shared %5zu  (x%d threads)\n", #K, fa.numRegs, fa.localSizeBytes, fa.sharedSizeBytes, T);
            QMMA_SATTR((qmma_q4_x<1, false>), 256) QMMA_SATTR((qmma_q4_x<4, false>), 512) QMMA_SATTR((qmma_q4_x<1, true>), 256)
            QMMA_SATTR((qmma_q8_0<1, true>), 256) QMMA_SATTR((qmma_q8_0<1, false>), 256) QMMA_SATTR(qmma_q4_K<1>, 128) QMMA_SATTR(qmma_q5_K<1>, 128)
            QMMA_SATTR(qmma_iq4_xs<1>, 256) QMMA_SATTR(qmma_iq3_s<1>, 256)
#undef QMMA_SATTR
        }
    }
    // split-K for narrow matrices: few tiles -> several warps per tile (env GGML_MUSA_QMMA_KS forces 1/2/4)
    static const int ks_env = getenv("GGML_MUSA_QMMA_KS") ? atoi(getenv("GGML_MUSA_QMMA_KS")) : 0;
    static const int nw_env = getenv("GGML_MUSA_QMMA_NW") ? atoi(getenv("GGML_MUSA_QMMA_NW")) : 0;
    // measured (N=4): R >= 65536 (LM head) best unsplit, 8192..65535 rows best 2-way, narrower 4-way
    // (ks, nw) per size class: [>= 4096 tiles (LM head), >= 512 tiles (>= 8192 rows), narrower]
    // env GGML_MUSA_QMMA_RULE="ks0,nw0,ks1,nw1,ks2,nw2" overrides (tuning)
    // measured end-to-end (MTP3): split only narrow matrices (< 8192 rows), 4 ways: 259.9 -> 252.3 ms/step
    static int rule[6] = {1, 0, 1, 0, 4, 0};
    static bool rule_init = false;
    if (!rule_init) {
        rule_init = true;
        if (const char * e = getenv("GGML_MUSA_QMMA_RULE")) {
            sscanf(e, "%d,%d,%d,%d,%d,%d", &rule[0], &rule[1], &rule[2], &rule[3], &rule[4], &rule[5]);
        }
    }
    const int cls = ntile >= 4096 ? 0 : ntile >= 512 ? 1 : 2;
    const int ks = ks_env > 0 ? ks_env : rule[2*cls];
    GGML_UNUSED(nw_env);   // warps per block follow from ks: base (1-2) for ks=1, 4 otherwise
    for (int64_t t0 = 0; t0 < N; t0 += QMMA_MAX_TOK) {
        const int nt = (int) std::min<int64_t>(QMMA_MAX_TOK, N - t0);
        const uint4  * yat = (const uint4 *)  (ya_p + (size_t) t0*ngrp*32);
        const float4 * dat = (const float4 *) (da_p + (size_t) t0*ngrp*4);
        const float4 * dlt = (const float4 *) (dl_p + (size_t) t0*ngrp*4);

        float * dt = (float *) dst->data + t0*sdst;
        switch (src0->type) {
            case GGML_TYPE_Q4_0: {
                if (ks == 1) {
                    qmma_q4_x<1, false><<<(ntile + 1)/2, 256, 0, st>>>((const uint8_t *) src0->data, yat, dat, dt, sdst, (int) R, nsb, nt, 0x01010101);
                } else if (ks == 2) {
                    qmma_q4_x<2, false><<<(ntile + 1)/2, 512, 0, st>>>((const uint8_t *) src0->data, yat, dat, dt, sdst, (int) R, nsb, nt, 0x01010101);
                } else {
                    qmma_q4_x<4, false><<<ntile, 512, 0, st>>>((const uint8_t *) src0->data, yat, dat, dt, sdst, (int) R, nsb, nt, 0x01010101);
                }
            } break;
            case GGML_TYPE_Q4_1:
                if (ks == 1) {
                    qmma_q4_x<1, true><<<(ntile + 1)/2, 256, 0, st>>>((const uint8_t *) src0->data, yat, dat, dt, sdst, (int) R, nsb, nt, 0x01010101);
                } else if (ks == 2) {
                    qmma_q4_x<2, true><<<(ntile + 1)/2, 512, 0, st>>>((const uint8_t *) src0->data, yat, dat, dt, sdst, (int) R, nsb, nt, 0x01010101);
                } else {
                    qmma_q4_x<4, true><<<ntile, 512, 0, st>>>((const uint8_t *) src0->data, yat, dat, dt, sdst, (int) R, nsb, nt, 0x01010101);
                }
                break;
            case GGML_TYPE_Q5_K:
                QMMA_KS_DISPATCH(qmma_q5_K, 1, ((const uint8_t *) src0->data, yat, dat, dt, sdst, (int) R, nsb, nt, 0x01010101));
                break;
            case GGML_TYPE_IQ4_XS:
                QMMA_KS_DISPATCH(qmma_iq4_xs, 2, ((const uint8_t *) src0->data, yat, dat, dt, sdst, (int) R, nsb, nt, 0x01010101));
                break;
            case GGML_TYPE_IQ3_S:
                QMMA_KS_DISPATCH(qmma_iq3_s, 2, ((const uint8_t *) src0->data, yat, dat, dt, sdst, (int) R, nsb, nt, 0x01010101));
                break;
            case GGML_TYPE_IQ3_XXS:
                QMMA_KS_DISPATCH(qmma_iq3_xxs, 2, ((const uint8_t *) src0->data, yat, dat, dt, sdst, (int) R, nsb, nt, 0x01010101));
                break;
            case GGML_TYPE_Q8_0: {
                // per-block exchange (V1) stays faster for the LM-head size class (>= 4096 tiles); V2 for the rest (~5%)
                static const int q8v1_env = getenv("GGML_MUSA_Q8_V1") ? atoi(getenv("GGML_MUSA_Q8_V1")) : -1;
                const bool q8v1 = q8v1_env >= 0 ? q8v1_env != 0 : cls == 0;
                const auto args = [&](auto kern, int grid, int thr) {
                    kern<<<grid, thr, 0, st>>>((const uint8_t *) src0->data, yat, dat, dt, sdst, (int) R, nsb, nt, 0x01010101);
                };
                if (ks == 1) {
                    q8v1 ? args(qmma_q8_0<1, false>, (ntile + 1)/2, 256) : args(qmma_q8_0<1, true>, (ntile + 1)/2, 256);
                } else if (ks == 2) {
                    q8v1 ? args(qmma_q8_0<2, false>, (ntile + 1)/2, 512) : args(qmma_q8_0<2, true>, (ntile + 1)/2, 512);
                } else {
                    q8v1 ? args(qmma_q8_0<4, false>, ntile, 512) : args(qmma_q8_0<4, true>, ntile, 512);
                }
            } break;
            case GGML_TYPE_Q4_K:
                QMMA_KS_DISPATCH(qmma_q4_K, 1, ((const uint8_t *) src0->data, yat, dat, dt, sdst, (int) R, nsb, nt, 0x01010101));
                break;
            case GGML_TYPE_IQ2_S:
                qmma_iq2_s<<<ntile, QMMA_IQ2S_SPLIT*128, 0, st>>>((const uint8_t *) src0->data, yat, dat, dlt, dt, sdst, (int) R, nsb, nt, 0x01010101);
                break;
            case GGML_TYPE_Q2_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: {
                const uint8_t * wp0 = (const uint8_t *) src0->data;
                if (src0->type == GGML_TYPE_Q2_K) {
                    QMMA_KS_DISPATCH(qmma_q2_K, 2, (wp0, yat, dat, dlt, dt, sdst, (int) R, nsb, nt, 0x01010101));
                } else if (src0->type == GGML_TYPE_Q6_K) {
                    QMMA_KS_DISPATCH(qmma_q6_K, 2, (wp0, yat, dat, dlt, dt, sdst, (int) R, nsb, nt, 0x01010101));
                } else if (src0->type == GGML_TYPE_IQ2_XXS) {
                    QMMA_KS_DISPATCH(qmma_iq2_xxs, 2, (wp0, yat, dat, dlt, dt, sdst, (int) R, nsb, nt, 0x01010101));
                } else {
                    QMMA_KS_DISPATCH(qmma_iq2_xs, 2, (wp0, yat, dat, dlt, dt, sdst, (int) R, nsb, nt, 0x01010101));
                }
            } break;
            default:
                GGML_ABORT("unsupported type");
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
