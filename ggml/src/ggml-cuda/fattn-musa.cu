// MTT S80 (MUSA mp_21) split-KV attention for small query batches (decode / speculative verify).
//
// ggml's generic FlashAttention kernels reach only ~50 GFLOPS on this GPU (tiny grids, 32-lane warp logic
// on 128-lane hardware). This kernel is the plain flash-decoding scheme:
//   pass 1: grid (KV chunk of 256, query token, KV head). 256 threads. Each thread scores one KV position
//           against the R = gqa query heads that share the KV head, the block computes the chunk's
//           softmax max/sum, then each thread accumulates one output dimension over the chunk.
//   pass 2: combine the chunk partials.
// Supported: D = 256 (K and V), f16 or q4_0 K/V of the same type, f16 mask, no softcap/ALiBi/sinks,
// gqa <= 8, a single sequence. Anything else returns false and falls back to the ggml kernels.

#include "common.cuh"
#include "fattn-musa.cuh"

#define FMUSA_D     256
#ifndef FMUSA_EXP
#define FMUSA_EXP 0   // timing experiments (experiments/fa/fa_exp_t.sh); 0 = normal
#endif
#define FMUSA_CHUNK 256

namespace {

// q4_0 value j of a 32-block: j < 16 -> low nibble of qs[j], else high nibble of qs[j-16]
template <int R, ggml_type type>
__launch_bounds__(FMUSA_CHUNK, 2)
__global__ void fmusa_pass1(const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
                            const half * __restrict__ mask, float * __restrict__ part_o, float2 * __restrict__ part_ml,
                            const float scale, const int n_kv, const int n_q, const int n_head,
                            const int64_t nbq1, const int64_t nbq2,
                            const int64_t nbk1, const int64_t nbk2, const int64_t nbv1, const int64_t nbv2,
                            const int64_t mask_stride) {
    __shared__ float qs[FMUSA_D*R];               // Q (scaled), [d][r]; reused as P[pos][r] after scoring
    __shared__ float red[R][FMUSA_CHUNK/32 + 1];

    const int tid = threadIdx.x, chunk = blockIdx.x, t = blockIdx.y, hk = blockIdx.z;
    const int nchunk = gridDim.x;
    const int pos = chunk*FMUSA_CHUNK + tid;

#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float * qrow = (const float *) (Q + (int64_t) t*nbq1 + (int64_t) (hk*R + r)*nbq2);
        qs[tid*R + r] = scale*qrow[tid];
    }
    __syncthreads();

    float s[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        s[r] = 0.0f;
    }
    const bool valid = pos < n_kv;
    if (valid) {
        const char * krow = K + (int64_t) pos*nbk1 + (int64_t) hk*nbk2;
        if constexpr (type == GGML_TYPE_F16) {
#pragma unroll 2
            for (int d0 = 0; d0 < FMUSA_D; d0 += 8) {
                const uint4 raw = *(const uint4 *) (krow + 2*d0);
                const half2 * h = (const half2 *) &raw;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 kf = __half22float2(h[i]);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        s[r] += qs[(d0 + 2*i)*R + r]*kf.x + qs[(d0 + 2*i + 1)*R + r]*kf.y;
                    }
                }
            }
        } else {
            const block_q4_0 * kb = (const block_q4_0 *) krow;
#pragma unroll 1
            for (int b = 0; b < FMUSA_D/QK4_0; ++b) {
                const float dk = __half2float(kb[b].d);
                const uint16_t * q16 = (const uint16_t *) kb[b].qs;
                float sb[R];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    sb[r] = 0.0f;
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t w = q16[i];
                    const float k0 = (float) ((int) ( w        & 0xF) - 8), k1 = (float) ((int) ((w >>  8) & 0xF) - 8);
                    const float k2 = (float) ((int) ((w >>  4) & 0xF) - 8), k3 = (float) ((int) ((w >> 12) & 0xF) - 8);
                    const int j = 2*i;   // dims b*32 + j, j+1 (low nibbles) and b*32 + 16 + j, j+1 (high nibbles)
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        sb[r] += qs[(b*32 + j)*R + r]*k0 + qs[(b*32 + j + 1)*R + r]*k1 +
                                 qs[(b*32 + 16 + j)*R + r]*k2 + qs[(b*32 + 17 + j)*R + r]*k3;
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    s[r] += dk*sb[r];
                }
            }
        }
        const float mv = __half2float(mask[(int64_t) t*mask_stride + pos]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            s[r] += mv;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        s[r] = valid ? s[r] : -INFINITY;
    }

    // block max per r: 32-thread groups reduce serially through smem (no shuffles: they are emulated on mp_21)
    __syncthreads();                      // everyone done reading qs
    float * P = qs;                       // [pos][r]
#pragma unroll
    for (int r = 0; r < R; ++r) {
        P[tid*R + r] = s[r];
    }
    __syncthreads();
    if (tid < R*(FMUSA_CHUNK/32)) {
        const int r = tid % R, g = tid / R;
        float m = -INFINITY;
        for (int i = 0; i < 32; ++i) {
            m = fmaxf(m, P[(g*32 + i)*R + r]);
        }
        red[r][g] = m;
    }
    __syncthreads();
    float m[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float mm = -INFINITY;
#pragma unroll
        for (int g = 0; g < FMUSA_CHUNK/32; ++g) {
            mm = fmaxf(mm, red[r][g]);
        }
        m[r] = mm;
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float p = (m[r] == -INFINITY || s[r] == -INFINITY) ? 0.0f : expf(s[r] - m[r]);
        P[tid*R + r] = p;
    }
    __syncthreads();
    if (tid < R*(FMUSA_CHUNK/32)) {
        const int r = tid % R, g = tid / R;
        float l = 0.0f;
        for (int i = 0; i < 32; ++i) {
            l += P[(g*32 + i)*R + r];
        }
        red[r][g] = l;
    }
    __syncthreads();

    // P*V: thread = output dimension
    const int nvalid = min(FMUSA_CHUNK, n_kv - chunk*FMUSA_CHUNK);
    float o[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        o[r] = 0.0f;
    }
    const int d = tid;
    const char * vbase = V + (int64_t) (chunk*FMUSA_CHUNK)*nbv1 + (int64_t) hk*nbv2;
    if constexpr (type == GGML_TYPE_F16) {
#pragma unroll 4
        for (int i = 0; i < nvalid; ++i) {
            const float v = __half2float(((const half *) (vbase + (int64_t) i*nbv1))[d]);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                o[r] += P[i*R + r]*v;
            }
        }
    } else {
        const int b = d/QK4_0, j = d%QK4_0;
        const int byte = j & 15, shift = (j >> 4)*4;
#pragma unroll 4
        for (int i = 0; i < nvalid; ++i) {
            const block_q4_0 * vb = (const block_q4_0 *) (vbase + (int64_t) i*nbv1) + b;
            const float v = __half2float(vb->d)*(float) ((int) ((vb->qs[byte] >> shift) & 0xF) - 8);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                o[r] += P[i*R + r]*v;
            }
        }
    }

    // partials: [t][head][chunk]
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int64_t idx = ((int64_t) t*n_head + hk*R + r)*nchunk + chunk;
        part_o[idx*FMUSA_D + d] = o[r];
        if (tid == 0) {
            float l = 0.0f;
#pragma unroll
            for (int g = 0; g < FMUSA_CHUNK/32; ++g) {
                l += red[r][g];
            }
            part_ml[idx] = make_float2(m[r], l);
        }
    }
}


// Tiled variant for several query tokens: a block handles TQ query tokens x R heads (RT rows) against one 256-position
// chunk, so every dequantized K/V element feeds RT multiply-adds instead of R. Q and P are staged in two halves
// (128 dims / 128 positions) to keep shared memory at 128*RT floats.
#ifndef FMUSA_NPART
#define FMUSA_NPART 4   // Q/P staged in FMUSA_NPART parts: smaller shared memory -> more blocks per MP
#endif
#ifndef FMUSA_T_MINB
#define FMUSA_T_MINB 3   // blocks/MP for the tiled kernel (scores + running outputs: 48 live accumulators)
#endif
template <int R, int TQ, ggml_type type>
__launch_bounds__(FMUSA_CHUNK, FMUSA_T_MINB)
__global__ void fmusa_pass1_t(const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
                              const half * __restrict__ mask, float * __restrict__ part_o, float2 * __restrict__ part_ml,
                              const float scale, const int n_kv, const int n_q, const int n_head,
                              const int64_t nbq1, const int64_t nbq2,
                              const int64_t nbk1, const int64_t nbk2, const int64_t nbv1, const int64_t nbv2,
                              const int64_t mask_stride, const int chunk0, const int nchunk_tot, const int cpb = 1) {
    // block = range blockIdx.x of cpb chunks: online softmax across them, one partial per range (index chunk0 + range)
    // K/V/mask point at the segment starting at absolute position chunk0*FMUSA_CHUNK; partials use absolute chunks
    constexpr int RT = R*TQ, NP = FMUSA_NPART, HALF = FMUSA_CHUNK/NP;
    __shared__ float buf[HALF*RT];                   // Q half [d][rr], later P half [pos][rr]
    __shared__ float red[RT][FMUSA_CHUNK/32 + 1];
    __shared__ float mrow[RT], mrun[RT], lrun[RT], alpha[RT];

    const int tid = threadIdx.x, rng = blockIdx.x, t0 = blockIdx.y*TQ, hk = blockIdx.z;
    const int nchunk = nchunk_tot;
    const int nck = (n_kv + FMUSA_CHUNK - 1)/FMUSA_CHUNK;
    const int d = tid;
    float o[RT];
#pragma unroll
    for (int rr = 0; rr < RT; ++rr) {
        o[rr] = 0.0f;
    }
    if (tid < RT) {
        mrun[tid] = -INFINITY;
        lrun[tid] = 0.0f;
    }
#pragma unroll 1
    for (int chunk = rng*cpb; chunk < min((rng + 1)*cpb, nck); ++chunk) {
    const int pos = chunk*FMUSA_CHUNK + tid;
    const bool valid = pos < n_kv;
    const char * krow = K + (int64_t) pos*nbk1 + (int64_t) hk*nbk2;

    float s[RT];
#pragma unroll
    for (int rr = 0; rr < RT; ++rr) {
        s[rr] = 0.0f;
    }
    // ---- scores, in two halves of the head dimension
#pragma unroll 1
    for (int hd = 0; hd < NP; ++hd) {
        __syncthreads();
        for (int e = tid; e < HALF*RT; e += FMUSA_CHUNK) {
            const int dl = e / RT, rr = e % RT, tq = rr / R, r = rr % R;
            float v = 0.0f;
            if (FMUSA_EXP != 35 && t0 + tq < n_q) {
                const float * qrow = (const float *) (Q + (int64_t) (t0 + tq)*nbq1 + (int64_t) (hk*R + r)*nbq2);
                v = scale*qrow[hd*HALF + dl];
            }
            buf[e] = v;
        }
        __syncthreads();
        if (valid && FMUSA_EXP != 31) {
            if constexpr (type == GGML_TYPE_F16) {
#pragma unroll 2
                for (int d0 = 0; d0 < HALF; d0 += 8) {
                    const uint4 raw = *(const uint4 *) (krow + 2*(hd*HALF + d0));
                    const half2 * h = (const half2 *) &raw;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 kf = __half22float2(h[i]);
#pragma unroll
                        for (int rr = 0; rr < RT; ++rr) {
                            s[rr] += buf[(d0 + 2*i)*RT + rr]*kf.x + buf[(d0 + 2*i + 1)*RT + rr]*kf.y;
                        }
                    }
                }
            } else {
                const block_q4_0 * kb = (const block_q4_0 *) krow + hd*(HALF/QK4_0);
#pragma unroll 1
                for (int b = 0; b < HALF/QK4_0; ++b) {
                    const float dk = __half2float(kb[b].d);
                    const uint16_t * q16 = (const uint16_t *) kb[b].qs;
                    float sb[RT];
#pragma unroll
                    for (int rr = 0; rr < RT; ++rr) {
                        sb[rr] = 0.0f;
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const uint32_t w = q16[i];
                        const float k0 = (float) ((int) ( w        & 0xF) - 8), k1 = (float) ((int) ((w >>  8) & 0xF) - 8);
                        const float k2 = (float) ((int) ((w >>  4) & 0xF) - 8), k3 = (float) ((int) ((w >> 12) & 0xF) - 8);
                        const int j = 2*i;
#pragma unroll
                        for (int rr = 0; rr < RT; ++rr) {
                            sb[rr] += buf[(b*32 + j)*RT + rr]*k0 + buf[(b*32 + j + 1)*RT + rr]*k1 +
                                      buf[(b*32 + 16 + j)*RT + rr]*k2 + buf[(b*32 + 17 + j)*RT + rr]*k3;
                        }
                    }
#pragma unroll
                    for (int rr = 0; rr < RT; ++rr) {
                        s[rr] += dk*sb[rr];
                    }
                }
            }
        }
    }
#pragma unroll
    for (int tq = 0; tq < TQ; ++tq) {
        const bool ok = valid && t0 + tq < n_q;
        const float mv = ok ? __half2float(mask[(int64_t) (t0 + tq)*mask_stride + pos]) : 0.0f;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            s[tq*R + r] = ok ? s[tq*R + r] + mv : -INFINITY;
        }
    }
    // ---- row max over the 256 positions, in two halves of positions
#pragma unroll 1
    for (int ph = 0; ph < NP; ++ph) {
        __syncthreads();
        if (tid / HALF == ph) {
#pragma unroll
            for (int rr = 0; rr < RT; ++rr) {
                buf[(tid & (HALF - 1))*RT + rr] = s[rr];
            }
        }
        __syncthreads();
        for (int e = tid; e < (FMUSA_EXP == 34 ? 0 : RT*(HALF/32)); e += FMUSA_CHUNK) {
            const int rr = e % RT, g = e / RT;
            float m = -INFINITY;
            for (int i = 0; i < 32; ++i) {
                m = fmaxf(m, buf[(g*32 + i)*RT + rr]);
            }
            red[rr][ph*(HALF/32) + g] = m;
        }
    }
    __syncthreads();
    if (tid < RT) {
        float mm = -INFINITY;
        for (int g = 0; g < FMUSA_CHUNK/32; ++g) {
            mm = fmaxf(mm, red[tid][g]);
        }
        const float mo = mrun[tid], mn = fmaxf(mo, mm);
        alpha[tid] = mo == -INFINITY ? 0.0f : expf(mo - mn);   // rescale of what was accumulated so far
        mrow[tid] = mn;
        mrun[tid] = mn;
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < RT; ++rr) {
        const float m = mrow[rr];
        s[rr] = (m == -INFINITY || s[rr] == -INFINITY) ? 0.0f : expf(s[rr] - m);
    }
    // ---- P*V (thread = output dimension) and row sums, in two halves of positions
    const int nvalid = min(FMUSA_CHUNK, n_kv - chunk*FMUSA_CHUNK);
    const char * vbase = V + (int64_t) (chunk*FMUSA_CHUNK)*nbv1 + (int64_t) hk*nbv2;
#pragma unroll
    for (int rr = 0; rr < RT; ++rr) {
        o[rr] *= alpha[rr];
    }
#pragma unroll 1
    for (int ph = 0; ph < NP; ++ph) {
        __syncthreads();
        if (tid / HALF == ph) {
#pragma unroll
            for (int rr = 0; rr < RT; ++rr) {
                buf[(tid & (HALF - 1))*RT + rr] = s[rr];
            }
        }
        __syncthreads();
        for (int e = tid; e < RT*(HALF/32); e += FMUSA_CHUNK) {
            const int rr = e % RT, g = e / RT;
            float l = 0.0f;
            for (int i = 0; i < 32; ++i) {
                l += buf[(g*32 + i)*RT + rr];
            }
            red[rr][ph*(HALF/32) + g] = l;
        }
        const int i0 = ph*HALF, i1 = min(nvalid, i0 + HALF);
        if constexpr (type == GGML_TYPE_F16) {
#pragma unroll 2
            for (int i = i0; i < (FMUSA_EXP == 32 ? i0 : i1); ++i) {
                const float v = __half2float(((const half *) (vbase + (int64_t) i*nbv1))[d]);
                const float * pr = buf + (i - i0)*RT;
#pragma unroll
                for (int rr = 0; rr < RT; ++rr) {
                    o[rr] += pr[rr]*v;
                }
            }
        } else {
            const int b = d/QK4_0, j = d%QK4_0;
            const int byte = j & 15, shift = (j >> 4)*4;
#pragma unroll 2
            for (int i = i0; i < i1; ++i) {
                const block_q4_0 * vb = (const block_q4_0 *) (vbase + (int64_t) i*nbv1) + b;
                const float v = __half2float(vb->d)*(float) ((int) ((vb->qs[byte] >> shift) & 0xF) - 8);
                const float * pr = buf + (i - i0)*RT;
#pragma unroll
                for (int rr = 0; rr < RT; ++rr) {
                    o[rr] += pr[rr]*v;
                }
            }
        }
    }
    __syncthreads();
    if (tid < RT) {
        float l = 0.0f;
        for (int g = 0; g < FMUSA_CHUNK/32; ++g) {
            l += red[tid][g];
        }
        lrun[tid] = lrun[tid]*alpha[tid] + l;
    }
    __syncthreads();
    }   // chunk loop
#pragma unroll
    for (int tq = 0; tq < TQ; ++tq) {
        if (t0 + tq >= n_q) {
            break;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int rr = tq*R + r;
            const int64_t idx = ((int64_t) (t0 + tq)*n_head + hk*R + r)*nchunk + chunk0 + rng;
            part_o[idx*FMUSA_D + d] = o[rr];
            if (tid == 0) {
                part_ml[idx] = make_float2(mrun[rr], lrun[rr]);
            }
        }
    }
}


// ---------------------------------------------------------------------------------------------------------------
// Tensor-core scores for few query tokens (MTP verify / drafts) with q4_0 K: S = K Q^T on the int8 MMA.
//   MMA m16n8k16 (unsigned): A = 16 positions x 16 dims of K (4-bit values n = q4_0 nibble, read straight from the cache:
//   one 32-bit word qs[4c..4c+3] gives dims 4c.. (low nibbles) and 16+4c.. (high nibbles)), B = 16 dims x 8 query rows
//   of Q quantized to int8 (+128), one scale per 32-dim block. A second MMA against all-ones gives every lane the
//   nibble sum of its row, so  sum_d K_d Q_d = dk*dq*(D - 128*sum(n) - 8*sum(q))  needs no cross-lane exchange.
//   Softmax works on the MMA layout; P*V is the FP32 loop of the tiled kernel (same partial format).
__device__ __forceinline__ int fmusa_mma2(const uint32_t a0v, const uint32_t a1v, const uint32_t b0v, const uint32_t b1v) {
    int a0[2] = {(int) a0v, 0}, a1[2] = {(int) a1v, 0};
    int b0[1] = {(int) b0v}, b1[1] = {(int) b1v};
    int c[4] = {0, 0, 0, 0}, d[4] = {0, 0, 0, 0};
    __musa_imma_m16n8k16_mma(d, a0, b0, c, 0, 0, 0, false, 0, 1);
    c[0] = d[0];
    __musa_imma_m16n8k16_mma(d, a1, b1, c, 0, 0, 0, false, 0, 1);
    return d[0];
}

// one MMA accumulating into c (int32 C input)
__device__ __forceinline__ int fmusa_mma_acc(const uint32_t av, const uint32_t bv, const int acc) {
    int a[2] = {(int) av, 0};
    int b[1] = {(int) bv};
    int c[4] = {acc, 0, 0, 0}, d[4] = {0, 0, 0, 0};
    __musa_imma_m16n8k16_mma(d, a, b, c, 0, 0, 0, false, 0, 1);
    return d[0];
}

// Q -> int8 MMA fragments. grid (n_head_kv, nslice), 256 threads; one thread per (row rr, 32-block b) (NG*8*8 <= 256).
//   qfrag[((hk*nslice + ts)*NG + g)*16 + kstep][32 lanes] u32,  kstep = 2b + h, lane = 4*col + c:  q[b*32 + 16h + 4c ..] ^ 0x80
//   qsc[((hk*nslice + ts)*NG*8 + rr)*8 + b] = (dq, -8*dq*sum(q))   (dq includes the softmax scale; 0 for padding rows)
template <int R, int TQ>
__global__ void fmusa_q_quant(const char * __restrict__ Q, uint32_t * __restrict__ qfrag, float2 * __restrict__ qsc,
                              const float scale, const int n_q, const int64_t nbq1, const int64_t nbq2) {
    constexpr int RT = R*TQ, NG = (RT + 7)/8;
    const int hk = blockIdx.x, ts = blockIdx.y, nslice = gridDim.y;
    const int i = threadIdx.x;
    if (i >= NG*8*8) {
        return;
    }
    const int rr = i / 8, b = i % 8, g = rr / 8, col = rr % 8;
    const int tq = rr / R, r = rr % R, t = ts*TQ + tq;
    const bool ok = rr < RT && t < n_q;
    float x[32];
    float amax = 0.0f;
    if (ok) {
        const float * qrow = (const float *) (Q + (int64_t) t*nbq1 + (int64_t) (hk*R + r)*nbq2) + b*32;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            x[k] = scale*qrow[k];
            amax = fmaxf(amax, fabsf(x[k]));
        }
    } else {
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            x[k] = 0.0f;
        }
    }
    const float dq = amax/127.0f, idq = amax > 0.0f ? 1.0f/dq : 0.0f;
    int sum = 0;
    uint32_t w[8];
#pragma unroll
    for (int k4 = 0; k4 < 8; ++k4) {
        uint32_t v = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int q = (int) roundf(x[4*k4 + e]*idq);
            sum += q;
            v |= (uint32_t) (uint8_t) (q + 128) << (8*e);
        }
        w[k4] = v;
    }
    const int64_t base = (int64_t) (hk*nslice + ts)*NG;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            qfrag[((base + g)*16 + 2*b + h)*32 + 4*col + c] = w[4*h + c];
        }
    }
    qsc[((int64_t) (hk*nslice + ts)*NG*8 + rr)*8 + b] = make_float2(dq, -8.0f*dq*(float) sum);
}

#ifndef FMUSA_MMA_MINB
#define FMUSA_MMA_MINB 4
#endif
// P*V on the tensor cores (PVM, default; GGML_MUSA_FA_PVMMA=0 -> FP32 loop): O[rr][d] = sum_pos P[rr][pos]*dv[pos][b]*(q[pos][d] - 8),
//   b = d/32. The V scale goes into P: u = round(P*dv/s_b) + 128 (int8 + 128), s_b = max |dv| of block b over the
//   chunk / 127, so the 4-bit q are used as they are (A = 16 dims x 16 positions, 4 positions per lane) and the MMA
//   sums stay int32 over the whole chunk:  O = s_b*(sum u*q - 128*sum q - 8*sum u + 8*128*256);
//   sum q (per dim) and sum u (per row) come from MMAs against all-ones B / A.
//   4 warps per block (64 positions of scores each, then 2 V blocks = 64 dims of P*V each): half the per-thread state
//   of a 2-warp block -> 12-16 warps per MP instead of 8 (the kernel is latency bound).
//   Per 16-position step: V staged in shared memory (coalesced 16-byte loads, prefetched one step ahead), the
//   owning warp writes its P (half) to shared memory, every warp quantizes B for its own V blocks, then all operand
//   loads, then the MMAs back to back (on mp_21 a load waits for every MMA in flight).
//   Accuracy (9B Q8_0, 16k window, 4-token batches, KLD vs FP32 P*V): 0.0035; for scale, f16 vs q4_0 KV 0.0041,
//   FP32 scores vs the int8-Q scores 0.0023. Tried and dropped: hi+lo bytes (spills, 2x slower), per-16-position
//   scales (KLD 0.0029, +25% time), unsigned P*|dv| with the sign moved to q (0.0030, +15% time).
#ifndef FMUSA_MMA_MINB
#define FMUSA_MMA_MINB 4
#endif
// P*V on the tensor cores (PVM): O[rr][d] = sum_pos P[rr][pos]*dv[pos][b]*(q[pos][d] - 8), b = d/32.
//   The V scale goes into P: u = round(P*dv/s_b) + 128 (int8 + 128), s_b = max |dv| of block b over the chunk / 127,
//   so the 4-bit q are used as they are (A = 16 dims x 16 positions, 4 positions per lane: one byte load each) and
//   the MMA sums stay int32 over the whole chunk:  O = s_b*(sum u*q - 128*sum q - 8*sum u + 8*128*256);
//   sum q (per dim) and sum u (per row) come from MMAs against all-ones B / A.
//   Each iteration: both warps quantize one 16-position step of their own scores into shared memory (B fragments,
//   [b][g][col][pos] bytes), then each warp runs both steps for its 4 V blocks (128 dims).
#ifndef FMUSA_PVM_MINB1
#define FMUSA_PVM_MINB1 4   // PVM blocks per MP, one 8-row group (1-token drafts): 62 registers, no spill
#endif
#ifndef FMUSA_PVM_MINBN
#define FMUSA_PVM_MINBN 3   // PVM, 2-4 row groups (verify): 4 blocks/MP spill heavily (3.67 vs 5.96 ms at 94k)
#endif
// warps per block: 2 (FP32 P*V: thread = output dim) or 4 (PVM: half the per-thread state -> 16 warps per MP)
template <bool PVM> constexpr int fmusa_mma_nw() { return PVM ? 4 : 2; }
template <int R, int TQ, bool PVM>
__launch_bounds__(128*fmusa_mma_nw<PVM>(), PVM ? (R*TQ <= 8 ? FMUSA_PVM_MINB1 : R*TQ <= 32 ? FMUSA_PVM_MINBN : 2) : FMUSA_MMA_MINB)
__global__ void fmusa_pass1_mma(const uint32_t * __restrict__ qfrag, const float2 * __restrict__ qsc,
                                const char * __restrict__ K, const char * __restrict__ V,
                                const half * __restrict__ mask, float * __restrict__ part_o, float2 * __restrict__ part_ml,
                                const int n_kv, const int n_q, const int n_head,
                                const int64_t nbk1, const int64_t nbk2, const int64_t nbv1, const int64_t nbv2,
                                const int64_t mask_stride, const int nchunk) {
    constexpr int RT = R*TQ, NG = (RT + 7)/8, NP = FMUSA_NPART, HALF = FMUSA_CHUNK/NP;
    constexpr int NW = fmusa_mma_nw<PVM>(), NT = 128*NW;
    constexpr int NRT = FMUSA_CHUNK/NW/16;           // 16-position row tiles per warp
    // P part [pos][rr]; during the softmax reductions the same memory holds red[NW][16][NG*8] (partial max / sum)
    constexpr int REDN = NW*16*NG*8;
    constexpr int BUFN0 = HALF*RT > REDN ? HALF*RT : REDN;
    constexpr int PVMN = (16*8*(int) sizeof(block_q4_0) + NG*8*16*2 + 8*NG*128)/4;   // Vs + Ps + Bs (PVM)
    constexpr int BUFN = PVM ? (PVMN > REDN ? PVMN : REDN) : BUFN0;
    __shared__ float buf[BUFN];
    float (*red)[16][NG*8] = (float (*)[16][NG*8]) buf;
    __shared__ float mrow[NG*8], lrow[NG*8];
    __shared__ float vsc[8], vinv[8];

    const int tid = threadIdx.x, l = tid & 127, wp = tid >> 7;
    const int chunk = blockIdx.x, ts = blockIdx.y, hk = blockIdx.z, nslice = gridDim.y;
    const int t0 = ts*TQ;
    const int row = l >> 3, col = l & 7;                       // D fragment
    const int arow = (l & 63) >> 2, ac = l & 3;                // A fragment (lanes 0-63)
    const int pbase = chunk*FMUSA_CHUNK + wp*(FMUSA_CHUNK/NW); // first position of this warp
    const uint32_t * qf = qfrag + (int64_t) (hk*nslice + ts)*NG*16*32 + (l & 31);
    const float2 * qs = qsc + (int64_t) (hk*nslice + ts)*NG*8*8;
    const char * kh = K + (int64_t) hk*nbk2;

    const uint32_t bones = l < 32 ? 0x01010101u : 0u;
    float sc[NRT][NG];
#pragma unroll
    for (int rt = 0; rt < NRT; ++rt) {
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            sc[rt][g] = 0.0f;
        }
    }
    // b outer: per 32-dim block, all operand loads of the warp's NRT row tiles first, then the NRT*(1 + NG) chained
    // MMA pairs back to back, then the FMAs (on mp_21 a load waits for every MMA in flight: interleaving loads with
    // MMAs serializes them)
    // 32-bit row offsets from the head base (64-bit pointers per row tile cost 2 registers each -> spills)
    const char * ka0 = kh + 2 + 4*ac;
    const int nbk1i = (int) nbk1;
    int kar[NRT], kdr[NRT];
#pragma unroll
    for (int rt = 0; rt < NRT; ++rt) {
        kar[rt] = min(pbase + rt*16 + arow, n_kv - 1)*nbk1i;   // A row position (clamped; masked later)
        kdr[rt] = min(pbase + rt*16 + row,  n_kv - 1)*nbk1i;   // D row position
    }
#pragma unroll 1
    for (int b = 0; b < FMUSA_D/QK4_0; ++b) {
        const int bo = b*(int) sizeof(block_q4_0);
        uint32_t qb0[NG], qb1[NG], wv[NRT];
        float2 qsv[NG];
        float dk[NRT];
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            qb0[g] = 0; qb1[g] = 0;
            if (l < 32) {                                                                   // B: lanes 0-31 only
                qb0[g] = __ldg(qf + (g*16 + 2*b)*32);
                qb1[g] = __ldg(qf + (g*16 + 2*b + 1)*32);
            }
            qsv[g] = __ldg(qs + (g*8 + col)*8 + b);
        }
#pragma unroll
        for (int rt = 0; rt < NRT; ++rt) {
            const uint16_t * q16 = (const uint16_t *) (ka0 + (kar[rt] + bo));
            wv[rt] = l < 64 ? ((uint32_t) q16[0] | ((uint32_t) q16[1] << 16)) : 0u;            // A: lanes 0-63 only
            dk[rt] = __half2float(*(const half *) (kh + (kdr[rt] + bo)));
        }
        int s1[NRT], D[NRT][NG];
#pragma unroll
        for (int rt = 0; rt < NRT; ++rt) {
            const uint32_t a0 = wv[rt] & 0x0F0F0F0Fu, a1 = (wv[rt] >> 4) & 0x0F0F0F0Fu;
            s1[rt] = fmusa_mma2(a0, a1, bones, bones);
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                D[rt][g] = fmusa_mma2(a0, a1, qb0[g], qb1[g]);
            }
        }
#pragma unroll
        for (int rt = 0; rt < NRT; ++rt) {
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                sc[rt][g] = fmaf(dk[rt], fmaf(qsv[g].x, (float) (D[rt][g] - 128*s1[rt]), qsv[g].y), sc[rt][g]);
            }
        }
    }
    // mask, invalid -> -inf; per-lane max over its NRT positions
    float pm[NG];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        pm[g] = -INFINITY;
        const int rr = g*8 + col, tq = rr / R;
        const bool rok = rr < RT && t0 + tq < n_q;
#pragma unroll
        for (int rt = 0; rt < NRT; ++rt) {
            const int pos = pbase + rt*16 + row;
            float v = -INFINITY;
            if (rok && pos < n_kv) {
                v = sc[rt][g] + __half2float(mask[(int64_t) (t0 + tq)*mask_stride + pos]);
            }
            sc[rt][g] = v;
            pm[g] = fmaxf(pm[g], v);
        }
        red[wp][row][g*8 + col] = pm[g];
    }
    __syncthreads();
    if (tid < NG*8) {
        float m = -INFINITY;
        for (int w = 0; w < NW; ++w) {
            for (int r2 = 0; r2 < 16; ++r2) {
                m = fmaxf(m, red[w][r2][tid]);
            }
        }
        mrow[tid] = m;
    }
    __syncthreads();
    float ps[NG];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const float m = mrow[g*8 + col];
        ps[g] = 0.0f;
#pragma unroll
        for (int rt = 0; rt < NRT; ++rt) {
            const float v = sc[rt][g];
            const float p = (m == -INFINITY || v == -INFINITY) ? 0.0f : expf(v - m);
            sc[rt][g] = p;
            ps[g] += p;
        }
    }
    __syncthreads();   // red reused
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        red[wp][row][g*8 + col] = ps[g];
    }
    __syncthreads();
    if (tid < NG*8) {
        float sum = 0.0f;
        for (int w = 0; w < NW; ++w) {
            for (int r2 = 0; r2 < 16; ++r2) {
                sum += red[w][r2][tid];
            }
        }
        lrow[tid] = sum;
    }
    const int nvalid = min(FMUSA_CHUNK, n_kv - chunk*FMUSA_CHUNK);
    if constexpr (PVM) {
    const char * vh = V + (int64_t) hk*nbv2;
    constexpr int VROW = 8*(int) sizeof(block_q4_0);       // 144 B: one position of this KV head
    // shared: Vs[16 pos][144 B] | Ps[NG*8 rows][16 pos] half | Bs[8 b][NG][8 col][16 pos] u8
    uint8_t * Vs = (uint8_t *) buf;
    half    * Ps = (half *) (Vs + 16*VROW);
    uint8_t * Bs = (uint8_t *) (Ps + NG*8*16);
    __syncthreads();   // lrow's red[] reads done before buf is reused
    constexpr int BO = 8/NW, NPG = NT/8, PPG = FMUSA_CHUNK/NPG;   // V blocks per warp; s_b pre-pass groups
    {   // s_b: max |dv| over the valid positions, thread = (b, PPG positions)
        const int b = tid & 7, pg = tid >> 3;
        float am = 0.0f;
#pragma unroll
        for (int k = 0; k < PPG; ++k) {
            const int p = pg*PPG + k;
            if (p < nvalid) {
                am = fmaxf(am, fabsf(__half2float(*(const half *) (vh + (int64_t) (chunk*FMUSA_CHUNK + p)*nbv1 + b*(int) sizeof(block_q4_0)))));
            }
        }
        buf[b*NPG + pg] = am;
    }
    __syncthreads();
    if (tid < 8) {
        float m = 0.0f;
        for (int k = 0; k < NPG; ++k) {
            m = fmaxf(m, buf[tid*NPG + k]);
        }
        vsc[tid]  = m/127.0f;
        vinv[tid] = m > 0.0f ? 127.0f/m : 0.0f;
    }
    int acc[BO][2][NG], aq[BO][2], au[BO][NG];
#pragma unroll
    for (int bo = 0; bo < BO; ++bo) {
        aq[bo][0] = 0; aq[bo][1] = 0;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            acc[bo][0][g] = 0; acc[bo][1][g] = 0; au[bo][g] = 0;
        }
    }
    const uint32_t bones = l < 32 ? 0x01010101u : 0u, aones = l < 64 ? 0x01010101u : 0u;
    // V of one 16-position step: 144 uint4, thread tid < 144 loads (pos tid/9, part tid%9); prefetched one step ahead
    const bool vld = tid < 16*VROW/16;
    const int vpos = tid / 9, vpart = tid % 9;
    auto vload = [&](const int p0) -> uint4 {
        uint4 v = make_uint4(0, 0, 0, 0);
        if (vld) {
            v = *(const uint4 *) (vh + (int64_t) min(p0 + vpos, n_kv - 1)*nbv1 + vpart*16);
        }
        return v;
    };
    uint4 vnext = vload(chunk*FMUSA_CHUNK);
#pragma unroll
    for (int it = 0; it < NRT; ++it) {
#pragma unroll 1
        for (int w2 = 0; w2 < NW; ++w2) {
            // step: positions chunk*256 + w2*256/NW + it*16 .., scores held by warp w2 (its row tile it)
            __syncthreads();   // previous step's reads of Vs/Ps/Bs done (first time: vsc/vinv visible)
            if (vld) {
                *(uint4 *) (Vs + vpos*VROW + vpart*16) = vnext;
            }
            {
                const int nit = w2 + 1 < NW ? it : it + 1, nw2 = w2 + 1 < NW ? w2 + 1 : 0;
                if (nit < NRT) {
                    vnext = vload(chunk*FMUSA_CHUNK + nw2*(FMUSA_CHUNK/NW) + nit*16);
                }
            }
            if (wp == w2) {
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    Ps[(g*8 + col)*16 + row] = __float2half(sc[it][g]);
                }
            }
            __syncthreads();
            {   // B bytes for this warp's BO V blocks: lane = (pos row, rr col)
                const half * vrow = (const half *) (Vs + row*VROW);
                float pv[NG];
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    pv[g] = __half2float(Ps[(g*8 + col)*16 + row]);
                }
#pragma unroll
                for (int bo = 0; bo < BO; ++bo) {
                    const int b = wp*BO + bo;
                    const float f = __half2float(vrow[b*(int) (sizeof(block_q4_0)/sizeof(half))])*vinv[b];
#pragma unroll
                    for (int g = 0; g < NG; ++g) {
                        Bs[((b*NG + g)*8 + col)*16 + row] = (uint8_t) __float_as_uint(fmaf(pv[g], f, 12583040.0f));   // 1.5*2^23 + 128
                    }
                }
            }
            __syncthreads();
            // all operand loads first, then the step's 4*(2 + 3*NG) MMAs back to back: on mp_21 a load waits for
            // every MMA in flight, so loads between MMAs serialize them (8 MMAs per load batch: 2.8 G/s, 32: 8.1 G/s)
            uint32_t alo[BO], ahi[BO], bw[BO][NG];
#pragma unroll
            for (int bo = 0; bo < BO; ++bo) {
                const int b = wp*BO + bo;
                alo[bo] = 0; ahi[bo] = 0;
                if (l < 64) {   // A: lanes 0-63 only. Row = byte arow of qs (dims arow, 16 + arow), k = positions 4ac..
                    const uint8_t * va = Vs + 4*ac*VROW + b*(int) sizeof(block_q4_0) + 2 + arow;
                    const uint32_t w = (uint32_t) va[0] | ((uint32_t) va[VROW] << 8) | ((uint32_t) va[2*VROW] << 16) | ((uint32_t) va[3*VROW] << 24);
                    alo[bo] = w & 0x0F0F0F0Fu;
                    ahi[bo] = (w >> 4) & 0x0F0F0F0Fu;
                }
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    bw[bo][g] = l < 32 ? *(const uint32_t *) (Bs + (b*NG + g)*128 + 4*l) : 0u;
                }
            }
#pragma unroll
            for (int bo = 0; bo < BO; ++bo) {
                aq[bo][0] = fmusa_mma_acc(alo[bo], bones, aq[bo][0]);
                aq[bo][1] = fmusa_mma_acc(ahi[bo], bones, aq[bo][1]);
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    acc[bo][0][g] = fmusa_mma_acc(alo[bo], bw[bo][g], acc[bo][0][g]);
                    acc[bo][1][g] = fmusa_mma_acc(ahi[bo], bw[bo][g], acc[bo][1][g]);
                    au[bo][g]     = fmusa_mma_acc(aones, bw[bo][g], au[bo][g]);
                }
            }
        }
    }
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const int rr = g*8 + col, tq = rr / R, r = rr % R;
        if (rr < RT && t0 + tq < n_q) {
            const int64_t idx = ((int64_t) (t0 + tq)*n_head + hk*R + r)*nchunk + chunk;
#pragma unroll
            for (int bo = 0; bo < BO; ++bo) {
                const int b = wp*BO + bo;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int v = acc[bo][h][g] - 128*aq[bo][h] - 8*au[bo][g] + 8*128*FMUSA_CHUNK;
                    part_o[idx*FMUSA_D + b*QK4_0 + 16*h + row] = vsc[b]*(float) v;
                }
            }
            if (row == 0 && wp == 0) {
                part_ml[idx] = make_float2(mrow[rr], lrow[rr]);
            }
        }
    }
    } else {
    // ---- P*V (thread = output dimension), P staged per part of HALF positions (the loop's first __syncthreads also
    // separates the red[] reads above from the buf writes)
    const int d = tid;
    const char * vbase = V + (int64_t) (chunk*FMUSA_CHUNK)*nbv1 + (int64_t) hk*nbv2;
    const int vb_i = d/QK4_0, vbyte = (d%QK4_0) & 15, vshift = ((d%QK4_0) >> 4)*4;
    float o[RT];
#pragma unroll
    for (int rr = 0; rr < RT; ++rr) {
        o[rr] = 0.0f;
    }
    constexpr int TPP = HALF/16;     // row tiles per part
#pragma unroll                       // static sc[][] indices (a runtime row-tile offset puts sc in local memory)
    for (int ph = 0; ph < NP; ++ph) {
        __syncthreads();
        // part ph = positions ph*HALF .. : warp ph*HALF/(CHUNK/2), its row tiles (ph*TPP) % NRT ..
        if (wp == (ph*HALF)/(FMUSA_CHUNK/2)) {
            const int rt0 = (ph*TPP) % NRT;
#pragma unroll
            for (int k = 0; k < TPP; ++k) {
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    const int rr = g*8 + col;
                    if (rr < RT) {
                        buf[(k*16 + row)*RT + rr] = sc[rt0 + k][g];
                    }
                }
            }
        }
        __syncthreads();
        const int i0 = ph*HALF, i1 = min(nvalid, i0 + HALF);
#pragma unroll 2
        for (int i = i0; i < i1; ++i) {
            const block_q4_0 * vb = (const block_q4_0 *) (vbase + (int64_t) i*nbv1) + vb_i;
            const float v = __half2float(vb->d)*(float) ((int) ((vb->qs[vbyte] >> vshift) & 0xF) - 8);
            const float * pr = buf + (i - i0)*RT;
#pragma unroll
            for (int rr = 0; rr < RT; ++rr) {
                o[rr] += pr[rr]*v;
            }
        }
    }
#pragma unroll
    for (int tq = 0; tq < TQ; ++tq) {
        if (t0 + tq >= n_q) {
            break;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int rr = tq*R + r;
            const int64_t idx = ((int64_t) (t0 + tq)*n_head + hk*R + r)*nchunk + chunk;
            part_o[idx*FMUSA_D + d] = o[rr];
            if (tid == 0) {
                part_ml[idx] = make_float2(mrow[rr], lrow[rr]);
            }
        }
    }
    }
}

// q4_0 K or V segment -> contiguous f16 [pos][head][256] (the tiled kernel then runs its f16 path: unpacking 4-bit
// values costs ~4 integer ops each on mp_21, as much as the attention math itself when done per query tile)
__global__ void fmusa_dequant_q4_0(const char * __restrict__ src, half * __restrict__ dst, const int npos, const int nhead,
                                   const int64_t nb1, const int64_t nb2) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;   // one thread per 32-value block
    const int64_t nblk = (int64_t) npos*nhead*(FMUSA_D/QK4_0);
    if (i >= nblk) {
        return;
    }
    const int b = (int) (i % (FMUSA_D/QK4_0)), h = (int) ((i / (FMUSA_D/QK4_0)) % nhead), p = (int) (i / ((FMUSA_D/QK4_0)*nhead));
    const block_q4_0 * blk = (const block_q4_0 *) (src + (int64_t) p*nb1 + (int64_t) h*nb2) + b;
    const float d = __half2float(blk->d);
    half2 * o = (half2 *) (dst + (((int64_t) p*nhead + h)*FMUSA_D + b*QK4_0));
#pragma unroll
    for (int j = 0; j < 16; j += 2) {
        const uint8_t q0 = blk->qs[j], q1 = blk->qs[j + 1];
        o[j/2]     = __floats2half2_rn(d*(float) ((int) (q0 & 0xF) - 8), d*(float) ((int) (q1 & 0xF) - 8));
        o[8 + j/2] = __floats2half2_rn(d*(float) ((int) (q0 >> 4) - 8), d*(float) ((int) (q1 >> 4) - 8));
    }
}

// pass 2: grid (head, token), FMUSA_CG*256 threads: thread = (chunk group, output dim). The chunk weights
// exp(m_c - M) are computed once per block into shared memory (FMUSA_CW at a time); each group sums every
// FMUSA_CG-th chunk with 8 loads in flight, then the groups are added. (Was: 256 threads looping over all chunks:
// 0.41 ms with 368 chunks and 24 heads, latency bound.)
#define FMUSA_CG 4
#define FMUSA_CW 1024
__launch_bounds__(FMUSA_CG*FMUSA_D, 1)
__global__ void fmusa_combine(const float * __restrict__ part_o, const float2 * __restrict__ part_ml, float * __restrict__ dst,
                              const int nchunk, const int n_head) {
    constexpr int NT = FMUSA_CG*FMUSA_D;
    __shared__ float wsh[FMUSA_CW];
    __shared__ float red[NT/32];
    __shared__ float osum[FMUSA_CG - 1][FMUSA_D];
    const int t = blockIdx.y, h = blockIdx.x, tid = threadIdx.x, d = tid % FMUSA_D, cg = tid / FMUSA_D;
    const int64_t base = ((int64_t) t*n_head + h)*nchunk;
    // M = max over chunks
    float m = -INFINITY;
    for (int c = tid; c < nchunk; c += NT) {
        m = fmaxf(m, part_ml[base + c].x);
    }
    // block max: each 32-thread slice -> red, then thread 0
    {
        __shared__ float mtmp[NT];
        mtmp[tid] = m;
        __syncthreads();
        if (tid < NT/32) {
            float x = -INFINITY;
            for (int k = 0; k < 32; ++k) {
                x = fmaxf(x, mtmp[tid*32 + k]);
            }
            red[tid] = x;
        }
        __syncthreads();
    }
    float M = -INFINITY;
    for (int k = 0; k < NT/32; ++k) {
        M = fmaxf(M, red[k]);
    }
    float num = 0.0f, den = 0.0f;
    for (int c0 = 0; c0 < nchunk; c0 += FMUSA_CW) {
        const int nc = min(FMUSA_CW, nchunk - c0);
        __syncthreads();   // previous wsh tile consumed
        for (int c = tid; c < nc; c += NT) {
            const float2 ml = part_ml[base + c0 + c];
            wsh[c] = (M == -INFINITY || ml.x == -INFINITY) ? 0.0f : expf(ml.x - M);
            den += wsh[c]*ml.y;   // per-thread partial; summed below
        }
        __syncthreads();
        const float * po = part_o + (base + c0)*FMUSA_D + d;
        int c = cg;
        for (; c + 7*FMUSA_CG < nc; c += 8*FMUSA_CG) {
            float x[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                x[k] = po[(int64_t) (c + k*FMUSA_CG)*FMUSA_D];
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                num = fmaf(wsh[c + k*FMUSA_CG], x[k], num);
            }
        }
        for (; c < nc; c += FMUSA_CG) {
            num = fmaf(wsh[c], po[(int64_t) c*FMUSA_D], num);
        }
    }
    // den: block sum of the per-thread partials; num: sum over chunk groups
    __syncthreads();
    {
        __shared__ float dtmp[NT];
        dtmp[tid] = den;
        if (cg > 0) {
            osum[cg - 1][d] = num;
        }
        __syncthreads();
        if (tid < NT/32) {
            float x = 0.0f;
            for (int k = 0; k < 32; ++k) {
                x += dtmp[tid*32 + k];
            }
            red[tid] = x;
        }
        __syncthreads();
    }
    if (cg == 0) {
        float D = 0.0f;
        for (int k = 0; k < NT/32; ++k) {
            D += red[k];
        }
#pragma unroll
        for (int k = 0; k < FMUSA_CG - 1; ++k) {
            num += osum[k][d];
        }
        dst[((int64_t) t*n_head + h)*FMUSA_D + d] = D > 0.0f ? num/D : 0.0f;
    }
}

// running (m, l, o) per (token, head) across KV segments: fold one segment's chunk partials in
__launch_bounds__(FMUSA_D, 4)
__global__ void fmusa_merge(const float * __restrict__ part_o, const float2 * __restrict__ part_ml,
                            float * __restrict__ run_o, float2 * __restrict__ run_ml, const int nc, const int n_head, const int first) {
    const int t = blockIdx.y, h = blockIdx.x, d = threadIdx.x;
    const int64_t base = ((int64_t) t*n_head + h)*nc, ri = (int64_t) t*n_head + h;
    const float2 rml = first ? make_float2(-INFINITY, 0.0f) : run_ml[ri];
    float M = rml.x;
    for (int c = 0; c < nc; ++c) {
        M = fmaxf(M, part_ml[base + c].x);
    }
    float num = 0.0f, den = 0.0f;
    if (M != -INFINITY) {
        if (rml.x != -INFINITY) {
            const float f = expf(rml.x - M);
            num = f*run_o[ri*FMUSA_D + d];
            den = f*rml.y;
        }
        for (int c = 0; c < nc; ++c) {
            const float2 ml = part_ml[base + c];
            if (ml.x == -INFINITY) {
                continue;
            }
            const float f = expf(ml.x - M);
            num += f*part_o[(base + c)*FMUSA_D + d];
            den += f*ml.y;
        }
    }
    run_o[ri*FMUSA_D + d] = num;
    __syncthreads();   // every thread has read run_ml[ri]
    if (d == 0) {
        run_ml[ri] = make_float2(M, den);
    }
}

__launch_bounds__(FMUSA_D, 4)
__global__ void fmusa_finalize(const float * __restrict__ run_o, const float2 * __restrict__ run_ml, float * __restrict__ dst, const int n_head) {
    const int t = blockIdx.y, h = blockIdx.x, d = threadIdx.x;
    const int64_t ri = (int64_t) t*n_head + h;
    const float l = run_ml[ri].y;
    dst[ri*FMUSA_D + d] = l > 0.0f ? run_o[ri*FMUSA_D + d]/l : 0.0f;
}

// GGML_MUSA_FA_PROF=1: event-timed phases (synchronizes; debugging only). Prints means every 50 calls.
struct fmusa_prof {
    bool on = getenv("GGML_MUSA_FA_PROF") != nullptr;
    musaEvent_t ev[4];
    double acc[3] = {0, 0, 0};
    long n = 0;
    fmusa_prof() { if (on) { for (auto & e : ev) { musaEventCreate(&e); } } }
    void mark(int i, cudaStream_t st) { if (on) { musaEventRecord(ev[i], st); } }
    void done() {
        if (!on) { return; }
        musaEventSynchronize(ev[3]);
        for (int i = 0; i < 3; ++i) { float ms; musaEventElapsedTime(&ms, ev[i], ev[i + 1]); acc[i] += ms; }
        if (++n % 50 == 0) {
            fprintf(stderr, "fmusa prof: q_quant %.1f us, pass1 %.1f us, combine %.1f us\n", 1e3*acc[0]/n, 1e3*acc[1]/n, 1e3*acc[2]/n);
        }
    }
};
static fmusa_prof & fmusa_prof_get() { static fmusa_prof p; return p; }

template <int R, int TQ, bool PVM>
void fmusa_mma_launch(ggml_backend_cuda_context & ctx, const char * qp, const ggml_tensor * K, const ggml_tensor * V, const half * mp,
                      float * po, float2 * pml, const float scale, const int n_kv, const int nt, const int n_head,
                      const int64_t nbq1, const int64_t nbq2, const int64_t mstride, const int nchunk) {
    constexpr int RT = R*TQ, NG = (RT + 7)/8;
    static_assert(NG*64 <= 512, "fmusa_q_quant: one thread per (row, block)");
    const int n_head_kv = (int) K->ne[2];
    const int nsl = (nt + TQ - 1)/TQ;
    cudaStream_t st = ctx.stream();
    ggml_cuda_pool_alloc<uint32_t> qfrag(ctx.pool(), (size_t) n_head_kv*nsl*NG*16*32);
    ggml_cuda_pool_alloc<float2>   qsc(ctx.pool(), (size_t) n_head_kv*nsl*NG*8*8);
    fmusa_prof_get().mark(0, st);
    fmusa_q_quant<R, TQ><<<dim3(n_head_kv, nsl), NG*64 > 256 ? 512 : 256, 0, st>>>(qp, qfrag.get(), qsc.get(), scale, nt, nbq1, nbq2);
    fmusa_prof_get().mark(1, st);
    {
        static bool once = false;
        if (!once && getenv("GGML_MUSA_FA_ATTR")) {
            once = true;
            musaFuncAttributes fa;
            musaFuncGetAttributes(&fa, (const void *) fmusa_pass1_mma<R, TQ, PVM>);
            fprintf(stderr, "fmusa_pass1_mma<%d,%d>: regs %d local %zu shared %zu\n", R, TQ, fa.numRegs, fa.localSizeBytes, fa.sharedSizeBytes);
            musaFuncGetAttributes(&fa, (const void *) fmusa_pass1_t<R, TQ, GGML_TYPE_Q4_0>);
            fprintf(stderr, "fmusa_pass1_t<%d,%d,q4_0>: regs %d local %zu shared %zu\n", R, TQ, fa.numRegs, fa.localSizeBytes, fa.sharedSizeBytes);
        }
    }
    fmusa_pass1_mma<R, TQ, PVM><<<dim3(nchunk, nsl, n_head_kv), 128*fmusa_mma_nw<PVM>(), 0, st>>>(qfrag.get(), qsc.get(),
        (const char *) K->data, (const char *) V->data, mp, po, pml, n_kv, nt, n_head,
        K->nb[1], K->nb[2], V->nb[1], V->nb[2], mstride, nchunk);
}

#define FMUSA_QSLICE 64   // query tokens per pass-1 launch (bounds the partial-result buffer)
#define FMUSA_SEG 16384   // KV positions per tiled launch (bounds the f16 scratch for q4_0 K/V: 2 x 32 MB with 4 KV heads)

template <int R, ggml_type type>
void fmusa_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V,
                  const ggml_tensor * mask, ggml_tensor * dst, const float scale) {
    const int n_kv = (int) K->ne[1], n_q = (int) Q->ne[1], n_head = (int) Q->ne[2], n_head_kv = (int) K->ne[2];
    const int nchunk = (n_kv + FMUSA_CHUNK - 1)/FMUSA_CHUNK;
    // partial results take nslice*n_head*nchunk*D floats: keep them under ~64 MB (at 128k context one query token
    // already needs 12 MB with 24 heads), otherwise long-context prompt processing runs out of VRAM
    const int64_t per_tok = (int64_t) n_head*nchunk*(FMUSA_D + 2)*sizeof(float);
    const int qslice = (int) std::max<int64_t>(1, std::min<int64_t>(FMUSA_QSLICE, (64ll << 20)/per_tok));
    const int nslice = std::min(n_q, qslice);
    cudaStream_t st = ctx.stream();
    const int64_t mstride = mask->nb[1]/sizeof(half);
    static const bool tiled_on = getenv("GGML_MUSA_FA_TILED") == nullptr || atoi(getenv("GGML_MUSA_FA_TILED")) != 0;
    constexpr int TQ = R >= 16 ? 1 : R >= 8 ? 3 : 4;   // RT = R*TQ <= 24 rows per block
    // experiment: GGML_MUSA_FA_MMA_PREFILL=1 sends q4_0 prompt batches at depth through the tensor-core kernel
    static const bool mma_prefill = getenv("GGML_MUSA_FA_MMA_PREFILL") != nullptr && atoi(getenv("GGML_MUSA_FA_MMA_PREFILL")) != 0;
    const bool prefill_to_mma = mma_prefill && type == GGML_TYPE_Q4_0 && R*TQ <= 32 && nchunk >= 32 && (int64_t) n_kv*K->nb[1] < INT32_MAX;
    static const int prefill_tq = getenv("GGML_MUSA_FA_MMA_PREFILL_TQ") ? atoi(getenv("GGML_MUSA_FA_MMA_PREFILL_TQ")) : 4;
    if (tiled_on && n_q >= 16 && TQ > 1 && !prefill_to_mma) {
        // Tiled path: KV in segments; each segment is (for q4_0 with many queries) unpacked to f16 once, every query
        // slice runs against it, and the segment's partials are folded into running (m, l, o) accumulators.
        const bool deq = type == GGML_TYPE_Q4_0 && n_q >= 16;
        const int seg_chunks = FMUSA_SEG/FMUSA_CHUNK;
        const int nseg_chunks = std::min(seg_chunks, nchunk);
        const int64_t per_tok_seg = (int64_t) n_head*nseg_chunks*(FMUSA_D + 2)*sizeof(float);
        const int qs = (int) std::max<int64_t>(TQ, std::min<int64_t>(n_q, (64ll << 20)/per_tok_seg)/TQ*TQ);
        ggml_cuda_pool_alloc<float>  po2(ctx.pool(), (size_t) qs*n_head*nseg_chunks*FMUSA_D);
        ggml_cuda_pool_alloc<float2> pml2(ctx.pool(), (size_t) qs*n_head*nseg_chunks);
        ggml_cuda_pool_alloc<float>  ro(ctx.pool(), (size_t) n_q*n_head*FMUSA_D);
        ggml_cuda_pool_alloc<float2> rml(ctx.pool(), (size_t) n_q*n_head);
        ggml_cuda_pool_alloc<half> kf, vf;
        if (deq) {
            const size_t seg_elems = (size_t) std::min(FMUSA_SEG, n_kv)*n_head_kv*FMUSA_D;
            kf.alloc(ctx.pool(), seg_elems);
            vf.alloc(ctx.pool(), seg_elems);
        }
        for (int c0 = 0; c0 < nchunk; c0 += seg_chunks) {
            const int nc = std::min(seg_chunks, nchunk - c0);
            const int p0 = c0*FMUSA_CHUNK, np = std::min(FMUSA_SEG, n_kv - p0);
            const char * kseg = (const char *) K->data + (int64_t) p0*K->nb[1];
            const char * vseg = (const char *) V->data + (int64_t) p0*V->nb[1];
            int64_t nbk1 = K->nb[1], nbk2 = K->nb[2], nbv1 = V->nb[1], nbv2 = V->nb[2];
            if (deq) {
                const int64_t nblk = (int64_t) np*n_head_kv*(FMUSA_D/QK4_0);
                if (FMUSA_EXP != 33) {
                fmusa_dequant_q4_0<<<(nblk + 255)/256, 256, 0, st>>>(kseg, kf.get(), np, n_head_kv, K->nb[1], K->nb[2]);
                fmusa_dequant_q4_0<<<(nblk + 255)/256, 256, 0, st>>>(vseg, vf.get(), np, n_head_kv, V->nb[1], V->nb[2]);
                }
                kseg = (const char *) kf.get(); vseg = (const char *) vf.get();
                nbk1 = nbv1 = (int64_t) n_head_kv*FMUSA_D*sizeof(half);
                nbk2 = nbv2 = FMUSA_D*sizeof(half);
            }
            // cpb chunks per block (online softmax inside the block): fewer partials to write and merge
            static const int cpb_env = getenv("GGML_MUSA_FA_CPB") ? std::max(1, atoi(getenv("GGML_MUSA_FA_CPB"))) : 4;
            const int cpb = std::min(cpb_env, nc), nr = (nc + cpb - 1)/cpb;
            for (int t0 = 0; t0 < n_q; t0 += qs) {
                const int nt = std::min(qs, n_q - t0);
                const dim3 grid(nr, (nt + TQ - 1)/TQ, n_head_kv);
                const char * qp = (const char *) Q->data + (int64_t) t0*Q->nb[1];
                const half * mseg = (const half *) mask->data + (int64_t) t0*mstride + p0;
                if (deq || type == GGML_TYPE_F16) {
                    fmusa_pass1_t<R, TQ, GGML_TYPE_F16><<<grid, FMUSA_CHUNK, 0, st>>>(qp, kseg, vseg, mseg, po2.get(), pml2.get(), scale,
                        np, nt, n_head, Q->nb[1], Q->nb[2], nbk1, nbk2, nbv1, nbv2, mstride, 0, nr, cpb);
                } else {
                    fmusa_pass1_t<R, TQ, type><<<grid, FMUSA_CHUNK, 0, st>>>(qp, kseg, vseg, mseg, po2.get(), pml2.get(), scale,
                        np, nt, n_head, Q->nb[1], Q->nb[2], nbk1, nbk2, nbv1, nbv2, mstride, 0, nr, cpb);
                }
                if (FMUSA_EXP != 36) fmusa_merge<<<dim3(n_head, nt), FMUSA_D, 0, st>>>(po2.get(), pml2.get(), ro.get() + (size_t) t0*n_head*FMUSA_D,
                    rml.get() + (size_t) t0*n_head, nr, n_head, c0 == 0);
            }
        }
        fmusa_finalize<<<dim3(n_head, n_q), FMUSA_D, 0, st>>>(ro.get(), rml.get(), (float *) dst->data, n_head);
        return;
    }
    ggml_cuda_pool_alloc<float>  po(ctx.pool(), (size_t) nslice*n_head*nchunk*FMUSA_D);
    ggml_cuda_pool_alloc<float2> pml(ctx.pool(), (size_t) nslice*n_head*nchunk);
    for (int t0 = 0; t0 < n_q; t0 += nslice) {
        const int nt = std::min(nslice, n_q - t0);
        // few query tokens (MTP verify): one tiled launch over all KV, no unpacking, no segments. Only with enough
        // KV chunks: at short context the per-token kernel's 4x more blocks fill the GPU better (measured 253 vs 258
        // ms per MTP step at ~300 positions, equal at 4k, tiled faster from 16k)
        static const bool mma_on = getenv("GGML_MUSA_FA_MMA") == nullptr || atoi(getenv("GGML_MUSA_FA_MMA")) != 0;
        // (the MMA kernel addresses K rows with 32-bit offsets)
        if (type == GGML_TYPE_Q4_0 && mma_on && R*TQ <= 32 && nchunk >= 32 && (int64_t) n_kv*K->nb[1] < INT32_MAX) {
            const char * qp = (const char *) Q->data + (int64_t) t0*Q->nb[1];
            const half * mp = (const half *) mask->data + (int64_t) t0*mstride;
            static const bool pvm = getenv("GGML_MUSA_FA_PVMMA") == nullptr || atoi(getenv("GGML_MUSA_FA_PVMMA")) != 0;
            if (nt == 1) {
                if (pvm) {
                    fmusa_mma_launch<R, 1, true>(ctx, qp, K, V, mp, po.get(), pml.get(), scale, n_kv, nt, n_head, Q->nb[1], Q->nb[2], mstride, nchunk);
                } else {
                    fmusa_mma_launch<R, 1, false>(ctx, qp, K, V, mp, po.get(), pml.get(), scale, n_kv, nt, n_head, Q->nb[1], Q->nb[2], mstride, nchunk);
                }
            } else if (prefill_to_mma && pvm && nt > TQ && prefill_tq == 8 && R*8 <= 48) {
                // prompt batches: 8 tokens per slice (48 rows for gqa 6) halve the K re-reads of 4-token slices
                constexpr int TQP = R*8 <= 48 ? 8 : TQ;
                fmusa_mma_launch<R, TQP, true>(ctx, qp, K, V, mp, po.get(), pml.get(), scale, n_kv, nt, n_head, Q->nb[1], Q->nb[2], mstride, nchunk);
            } else {
                if (pvm) {
                    fmusa_mma_launch<R, TQ, true>(ctx, qp, K, V, mp, po.get(), pml.get(), scale, n_kv, nt, n_head, Q->nb[1], Q->nb[2], mstride, nchunk);
                } else {
                    fmusa_mma_launch<R, TQ, false>(ctx, qp, K, V, mp, po.get(), pml.get(), scale, n_kv, nt, n_head, Q->nb[1], Q->nb[2], mstride, nchunk);
                }
            }
        } else if (tiled_on && nt > 1 && TQ > 1 && nchunk >= 32) {
            fmusa_pass1_t<R, TQ, type><<<dim3(nchunk, (nt + TQ - 1)/TQ, n_head_kv), FMUSA_CHUNK, 0, st>>>(
                (const char *) Q->data + (int64_t) t0*Q->nb[1], (const char *) K->data, (const char *) V->data,
                (const half *) mask->data + (int64_t) t0*mstride,
                po.get(), pml.get(), scale, n_kv, nt, n_head, Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mstride, 0, nchunk);
        } else {
        fmusa_pass1<R, type><<<dim3(nchunk, nt, n_head_kv), FMUSA_CHUNK, 0, st>>>(
            (const char *) Q->data + (int64_t) t0*Q->nb[1], (const char *) K->data, (const char *) V->data,
            (const half *) mask->data + (int64_t) t0*mstride,
            po.get(), pml.get(), scale, n_kv, nt, n_head, Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mstride);
        }
        fmusa_prof_get().mark(2, st);
        fmusa_combine<<<dim3(n_head, nt), FMUSA_CG*FMUSA_D, 0, st>>>(po.get(), pml.get(),
            (float *) dst->data + (int64_t) t0*n_head*FMUSA_D, nchunk, n_head);
        fmusa_prof_get().mark(3, st);
        fmusa_prof_get().done();
    }
}

template <ggml_type type>
bool fmusa_dispatch_r(ggml_backend_cuda_context & ctx, const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V,
                      const ggml_tensor * mask, ggml_tensor * dst, const float scale, const int gqa) {
    switch (gqa) {
        case 1: fmusa_launch<1, type>(ctx, Q, K, V, mask, dst, scale); return true;
        case 2: fmusa_launch<2, type>(ctx, Q, K, V, mask, dst, scale); return true;
        case 4: fmusa_launch<4, type>(ctx, Q, K, V, mask, dst, scale); return true;
        case 6: fmusa_launch<6, type>(ctx, Q, K, V, mask, dst, scale); return true;
        case 8: fmusa_launch<8, type>(ctx, Q, K, V, mask, dst, scale); return true;
        case 16: fmusa_launch<16, type>(ctx, Q, K, V, mask, dst, scale); return true;
        default: return false;
    }
}

int fmusa_max_q() {
    static const int v = [] {
        const char * e = getenv("GGML_MUSA_FA_MAXQ");
        return e ? atoi(e) : 1 << 30;
    }();
    return v;
}

} // namespace

bool ggml_cuda_flash_attn_ext_musa_ok(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    // src[5] = kv_rows (attention over a list of KV cells): not handled here
    if (!Q || !K || !V || !mask || sinks || dst->src[5] || dst->src[6] || max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != V->type || (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q4_0) ||
        mask->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != FMUSA_D || K->ne[0] != FMUSA_D || V->ne[0] != FMUSA_D || Q->ne[3] != 1 || K->ne[3] != 1 ||
        V->ne[3] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1 || Q->nb[0] != sizeof(float)) {
        return false;
    }
    if (Q->ne[1] > fmusa_max_q() || K->ne[1] != V->ne[1] || Q->ne[2] % K->ne[2] != 0 || V->ne[2] != K->ne[2]) {
        return false;
    }
    const int gqa = (int) (Q->ne[2]/K->ne[2]);
    if (gqa != 1 && gqa != 2 && gqa != 4 && gqa != 6 && gqa != 8 && gqa != 16) {
        return false;
    }
    if (K->type == GGML_TYPE_F16 && (K->nb[1] % 16 != 0 || ((uintptr_t) K->data) % 16 != 0)) {
        return false;
    }
    if (!ggml_is_contiguous(dst)) {
        return false;
    }
    static const bool disabled = getenv("GGML_MUSA_FA_DISABLE") != nullptr && atoi(getenv("GGML_MUSA_FA_DISABLE")) != 0;
    return !disabled;
}

bool ggml_cuda_flash_attn_ext_musa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!ggml_cuda_flash_attn_ext_musa_ok(dst)) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const int gqa = (int) (Q->ne[2]/K->ne[2]);
    static const bool dbg = getenv("GGML_MUSA_FA_DEBUG") != nullptr;
    if (dbg) {
        fprintf(stderr, "fmusa: n_q=%d n_kv=%d gqa=%d type=%s\n", (int) Q->ne[1], (int) K->ne[1], gqa, ggml_type_name(K->type));
    }
    if (K->type == GGML_TYPE_F16) {
        return fmusa_dispatch_r<GGML_TYPE_F16>(ctx, Q, K, V, mask, dst, scale, gqa);
    }
    return fmusa_dispatch_r<GGML_TYPE_Q4_0>(ctx, Q, K, V, mask, dst, scale, gqa);
}
