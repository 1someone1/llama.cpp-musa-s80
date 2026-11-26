// MTT S80: graph-level fusions for chains of tiny element-wise ops (Inkling 2 Mini decode runs ~7800 kernels per
// token, most of them on a few hundred floats; each costs ~10-25 us on this GPU, far more than its work).
#include "musa-fuse.cuh"
#include "convert.cuh"
#include <initializer_list>
#include <string>
#include <vector>

// debug (timing by elimination): GGML_MUSA_SKIP_FUSED=ctrl,mix,conv,situ,relb,router,clip,mmgroup skips the kernels of
// those fusions (the nodes are still consumed; output is garbage)
bool musa_skip_fused(const char * name) {
    static const std::string list = [] { const char * e = getenv("GGML_MUSA_SKIP_FUSED"); return std::string(e ? e : ""); }();
    if (list.empty()) {
        return false;
    }
    const std::string n(name);
    size_t a = 0;
    while (a <= list.size()) {
        size_t b = list.find(',', a);
        if (b == std::string::npos) b = list.size();
        if (list.compare(a, b - a, n) == 0) {
            return true;
        }
        a = b + 1;
    }
    return false;
}

// A fused kernel writes its outputs at the position of the chain's first node, but the graph allocator planned their
// memory for their original (later) positions: an output may share memory with an input that is dead by then in the
// original order. Fusions check every written range against every read range and fall back on any overlap.

// small device-to-device copies as a kernel (cudaMemcpyAsync D2D is a slow DMA path for a few KB on MUSA)
static __global__ void musa_copy_u32(const uint32_t * __restrict__ src, uint32_t * __restrict__ dst, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}
static void musa_copy_small(cudaStream_t st, void * dst, const void * src, size_t nbytes) {
    if (nbytes % 4 != 0 || ((uintptr_t) dst % 4) != 0 || ((uintptr_t) src % 4) != 0) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, st));
        return;
    }
    const int n = (int) (nbytes/4);
    musa_copy_u32<<<(n + 255)/256, 256, 0, st>>>((const uint32_t *) src, (uint32_t *) dst, n);
}

static bool musa_overlap(const void * a, size_t na, const void * b, size_t nb) {
    const char * pa = (const char *) a, * pb = (const char *) b;
    return pa < pb + nb && pb < pa + na;
}
static bool musa_disjoint_impl(std::initializer_list<const ggml_tensor *> writes, std::initializer_list<const ggml_tensor *> reads, int line) {
    int wi = 0;
    for (const ggml_tensor * w : writes) {
        int ri = 0;
        for (const ggml_tensor * r : reads) {
            if (w && r && musa_overlap(w->data, ggml_nbytes(w), r->data, ggml_nbytes(r))) {
                static const bool dbg = getenv("GGML_MUSA_FUSE_DBG") != nullptr;
                static int n = 0;
                if (dbg && n++ < 40) {
                    fprintf(stderr, "musa_disjoint: line %d write %d (%s) overlaps read %d (%s)%s\n", line, wi, w->name, ri, r->name,
                            w->data == r->data ? " [same address]" : "");
                }
                return false;
            }
            ++ri;
        }
        ++wi;
    }
    return true;
}

// ---- row clip: Inkling's clip() = scale each row down to an L2 norm of at most c
//   SQR(x) -> SUM_ROWS -> SQRT -> SCALE(s, b) -> FILL(c) -> DIV(fill, scaled) -> CLAMP(lo, hi) -> MUL(x, factor)
//   out = x * clamp(c / (s*sqrt(sum x^2) + b), lo, hi), per row of ne0
// one block of BS threads per row; ne0 <= BS: each thread holds one element in a register, two-level shared-memory sum
template <int BS>
static __global__ void __launch_bounds__(BS) musa_row_clip(const float * __restrict__ x, float * __restrict__ dst, const int ne0,
        const int ne1, const int ne2, const int64_t sx1, const int64_t sx2, const int64_t sx3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3,
        const float s, const float b, const float c, const float lo, const float hi, const float * __restrict__ w = nullptr) {
    __shared__ float sq[BS];
    __shared__ float part[BS/8];
    const int r = blockIdx.x, i1 = r % ne1, i2 = (r / ne1) % ne2, i3 = r / (ne1*ne2);
    const int t = threadIdx.x;
    const float * xr = x + i1*sx1 + i2*sx2 + i3*sx3;
    float v = 0.0f;
    if (t < ne0) {
        v = xr[t];
        if (w) {                 // optional per-element scale (contiguous, same shape as x)
            v *= w[(int64_t) r*ne0 + t];
        }
    }
    sq[t] = v*v;
    __syncthreads();
    if (t < BS/8) {
        float a = 0.0f;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            a += sq[t*8 + u];
        }
        part[t] = a;
    }
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int u = 0; u < BS/8; ++u) {
        tot += part[u];
    }
    const float f = fminf(fmaxf(c / (s*sqrtf(tot) + b), lo), hi);
    if (t < ne0) {
        dst[i1*sd1 + i2*sd2 + i3*sd3 + t] = v*f;
    }
}

static bool musa_is_f32_rows(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float);
}

static int musa_fuse_row_clip(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    if (i + 7 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor ** n = cgraph->nodes + i;
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL,
                                             GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL }, { i + 7 })) {
        return 0;
    }
    const ggml_tensor * x = n[0]->src[0];
    if (n[1]->src[0] != n[0] || n[2]->src[0] != n[1] || n[3]->src[0] != n[2] || n[4]->src[0] != n[3] ||
        n[5]->src[0] != n[4] || n[5]->src[1] != n[3] || n[6]->src[0] != n[5] ||
        n[7]->src[0] != x || n[7]->src[1] != n[6]) {
        return 0;
    }
    ggml_tensor * dst = n[7];
    if (!musa_is_f32_rows(x) || !musa_is_f32_rows(dst) || !ggml_are_same_shape(x, dst) ||
        n[1]->ne[0] != 1 || !ggml_are_same_shape(n[6], n[1])) {
        return 0;
    }
    if (dst->data != x->data && !musa_disjoint_impl({ dst }, { x }, __LINE__)) {
        return 0;
    }
    const float s  = ggml_get_op_params_f32(n[3], 0), b = ggml_get_op_params_f32(n[3], 1);
    const float c  = ggml_get_op_params_f32(n[4], 0);
    const float lo = ggml_get_op_params_f32(n[6], 0), hi = ggml_get_op_params_f32(n[6], 1);
    const int64_t nrows = x->ne[1]*x->ne[2]*x->ne[3];
    if (nrows > INT32_MAX || x->ne[0] > 128) {     // one element per thread
        return 0;
    }
    const int64_t f = sizeof(float);
    if (musa_skip_fused("clip")) {
        return 7;
    }
    musa_row_clip<128><<<(unsigned) nrows, 128, 0, ctx.stream()>>>((const float *) x->data, (float *) dst->data, (int) x->ne[0],
        (int) x->ne[1], (int) x->ne[2], x->nb[1]/f, x->nb[2]/f, x->nb[3]/f, dst->nb[1]/f, dst->nb[2]/f, dst->nb[3]/f,
        s, b, c, lo, hi);
    return 7;
}

// ---- Inkling loop controller, control(): 10 nodes -> 2 kernels (decode, one token)
//   g   = W_down . concat(a, b)                     (a, b = the RMS-normed cur and message, 2 x K/2)
//   out = (x - tanh(W_forget . g)*x) + tanh(W_gain . g)*m        (x = cur, m = message, ne0 = n)
//   CONCAT, MUL_MAT, MUL_MAT, TANH, MUL, SUB, MUL_MAT, TANH, MUL, ADD; weights Q8_0 in the original layout
// kernel 1: one 128-lane warp per row of W_down (rank rows, K = 2*n), lane l takes 32-blocks l, l+128, ...
template <bool NORM>
static __global__ void musa_ctrl_down(const block_q8_0 * __restrict__ w, const float * __restrict__ a, const float * __restrict__ b,
        float * __restrict__ g, const int half_k, const int rank, const float eps_a, const float eps_b,
        const int64_t a_ts = 0, const int64_t b_ts = 0) {
    a += (int64_t) blockIdx.y*a_ts;          // token blockIdx.y
    b += (int64_t) blockIdx.y*b_ts;
    g += (int64_t) blockIdx.y*rank;
    __shared__ float red[4][128];
    __shared__ float r2[4][16];
    __shared__ float nrm[2];
    const int l = threadIdx.x & 127, wp = threadIdx.x >> 7;
    float sa = 1.0f, sb = 1.0f;
    if constexpr (NORM) {
        // every block normalizes both halves itself (2 x half_k floats): RMS_NORM, eps from the graph
        __shared__ float nr[2][512];
        float pa = 0.0f, pb = 0.0f;
        for (int k = threadIdx.x; k < half_k; k += 512) {
            pa = fmaf(a[k], a[k], pa);
            pb = fmaf(b[k], b[k], pb);
        }
        nr[0][threadIdx.x] = pa;
        nr[1][threadIdx.x] = pb;
        __syncthreads();
        if (threadIdx.x < 64) {
            const int c = threadIdx.x >> 5, p = threadIdx.x & 31;
            float t = 0.0f;
#pragma unroll
            for (int u = 0; u < 16; ++u) {
                t += nr[c][p*16 + u];
            }
            red[c][p] = t;
        }
        __syncthreads();
        if (threadIdx.x < 2) {
            float t = 0.0f;
#pragma unroll
            for (int u = 0; u < 32; ++u) {
                t += red[threadIdx.x][u];
            }
            nrm[threadIdx.x] = rsqrtf(t/half_k + (threadIdx.x ? eps_b : eps_a));
        }
        __syncthreads();
        sa = nrm[0];
        sb = nrm[1];
    }
    const int r = blockIdx.x*4 + wp;
    const int nb = 2*half_k/32;
    float acc = 0.0f;
    if (r < rank) {
        const block_q8_0 * wr = w + (int64_t) r*nb;
        for (int ib = l; ib < nb; ib += 128) {
            const int k0 = ib*32;
            const float * c = k0 < half_k ? a + k0 : b + (k0 - half_k);
            const float cs = k0 < half_k ? sa : sb;
            const int8_t * q = wr[ib].qs;
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                s = fmaf((float) q[j], NORM ? c[j]*cs : c[j], s);
            }
            acc = fmaf(__half2float(wr[ib].d), s, acc);
        }
    }
    red[wp][l] = acc;
    __syncthreads();
    if (l < 16) {
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s += red[wp][l*8 + j];
        }
        r2[wp][l] = s;
    }
    __syncthreads();
    if (l == 0 && r < rank) {
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            s += r2[wp][j];
        }
        g[r] = s;
    }
}

// kernel 2: 64 output rows per 256-thread block; 4 threads per row = (forget, gain) x (two 32-k halves of rank 64)
template <int RANK>
static __global__ void musa_ctrl_up(const block_q8_0 * __restrict__ wf, const block_q8_0 * __restrict__ wg,
        const float * __restrict__ g, const float * __restrict__ x, const float * __restrict__ m, float * __restrict__ out, const int n,
        const int64_t x_ts = 0, const int64_t m_ts = 0) {
    g += (int64_t) blockIdx.y*RANK;          // token blockIdx.y
    x += (int64_t) blockIdx.y*x_ts;
    m += (int64_t) blockIdx.y*m_ts;
    out += (int64_t) blockIdx.y*n;
    constexpr int NB = RANK/32;            // 32-blocks per row (2)
    constexpr int TPR = 2*NB;              // threads per row
    constexpr int RPB = 256/TPR;           // rows per block
    __shared__ float gs[RANK];
    __shared__ float part[RPB][TPR];
    for (int k = threadIdx.x; k < RANK; k += 256) {
        gs[k] = g[k];
    }
    __syncthreads();
    const int lr = threadIdx.x / TPR, sub = threadIdx.x % TPR;
    const int r = blockIdx.x*RPB + lr;
    const int which = sub / NB, ib = sub % NB;     // which: 0 forget, 1 gain
    float s = 0.0f;
    if (r < n) {
        const block_q8_0 & blk = (which ? wg : wf)[(int64_t) r*NB + ib];
        const float * gv = gs + ib*32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            s = fmaf((float) blk.qs[j], gv[j], s);
        }
        s *= __half2float(blk.d);
    }
    part[lr][sub] = s;
    __syncthreads();
    if (sub == 0 && r < n) {
        float f = 0.0f, ga = 0.0f;
#pragma unroll
        for (int j = 0; j < NB; ++j) {
            f  += part[lr][j];
            ga += part[lr][NB + j];
        }
        f = tanhf(f);
        ga = tanhf(ga);
        const float xr = x[r];
        out[r] = (xr - f*xr) + ga*m[r];
    }
}

static bool musa_is_f32_vec(const ggml_tensor * t, int64_t n) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == n;
}

static int musa_fuse_control(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = getenv("GGML_MUSA_FUSE_CTRL") == nullptr || atoi(getenv("GGML_MUSA_FUSE_CTRL")) != 0;
    if (!on || i + 9 >= cgraph->n_nodes) {
        return 0;
    }
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_CONCAT, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL,
                                             GGML_OP_SUB, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD }, { i + 9 })) {
        return 0;
    }
    ggml_tensor ** nd = cgraph->nodes + i;
    ggml_tensor * cat = nd[0], * down = nd[1], * mmf = nd[2], * tf = nd[3], * mf = nd[4], * sb = nd[5];
    ggml_tensor * mmg = nd[6], * tg = nd[7], * mg = nd[8], * out = nd[9];
    const ggml_tensor * a = cat->src[0], * b = cat->src[1];
    const ggml_tensor * x = sb->src[0];
    if (ggml_get_unary_op(tf) != GGML_UNARY_OP_TANH || ggml_get_unary_op(tg) != GGML_UNARY_OP_TANH) {
        return 0;
    }
    if (ggml_get_op_params_i32(cat, 0) != 0 || down->src[1] != cat || mmf->src[1] != down || mmg->src[1] != down ||
        tf->src[0] != mmf || mf->src[0] != tf || mf->src[1] != x || sb->src[1] != mf ||
        tg->src[0] != mmg || mg->src[0] != tg || out->src[0] != sb || out->src[1] != mg) {
        return 0;
    }
    const ggml_tensor * m = mg->src[1];
    const ggml_tensor * wd = down->src[0], * wf = mmf->src[0], * wg = mmg->src[0];
    const int64_t n = x->ne[0], half_k = a->ne[0], rank = wd->ne[1];
    if (wd->type != GGML_TYPE_Q8_0 || wf->type != GGML_TYPE_Q8_0 || wg->type != GGML_TYPE_Q8_0 ||
        !ggml_is_contiguous(wd) || !ggml_is_contiguous(wf) || !ggml_is_contiguous(wg) ||
        rank != 64 || wd->ne[0] != 2*half_k || half_k % 32 != 0 || b->ne[0] != half_k ||
        wf->ne[0] != rank || wg->ne[0] != rank || wf->ne[1] != n || wg->ne[1] != n ||
        !musa_is_f32_vec(a, half_k) || !musa_is_f32_vec(b, half_k) || !musa_is_f32_vec(x, n) || !musa_is_f32_vec(m, n) ||
        !musa_is_f32_vec(down, rank) || !musa_is_f32_vec(out, n) || n > INT32_MAX) {
        return 0;
    }
    // the weights must still be in the original layout (not tiled for the tensor-core path)
    if (ggml_cuda_q4_0_mma_is_tiled(wd) || ggml_cuda_q4_0_mma_is_tiled(wf) || ggml_cuda_q4_0_mma_is_tiled(wg)) {
        return 0;
    }
    if (!musa_disjoint_impl({ down, out }, { a, b, x, m, wd, wf, wg }, __LINE__) || musa_overlap(down->data, ggml_nbytes(down), out->data, ggml_nbytes(out))) {
        return 0;
    }
    cudaStream_t st = ctx.stream();
    if (musa_skip_fused("ctrl")) {
        return 9;
    }
    musa_ctrl_down<false><<<(unsigned) ((rank + 3)/4), 512, 0, st>>>((const block_q8_0 *) wd->data, (const float *) a->data,
        (const float *) b->data, (float *) down->data, (int) half_k, (int) rank, 0.0f, 0.0f);
    musa_ctrl_up<64><<<(unsigned) ((n + 63)/64), 256, 0, st>>>((const block_q8_0 *) wf->data, (const block_q8_0 *) wg->data,
        (const float *) down->data, (const float *) x->data, (const float *) m->data, (float *) out->data, (int) n);
    return 9;
}

// ---- Inkling depth attention, mix(): states = rows s_0..s_{k-1} (chain of CONCATs along dim 1), one token
//   CONCAT x (k-1), RMS_NORM, TRANSPOSE, CONT, REPEAT(query), MUL_MAT, SOFT_MAX, MUL_MAT  ->  2 kernels
//   out[:, q] = sum_j softmax_j(scale * (s_j/rms_j) . query_q) * s_j/rms_j
#define MUSA_MIX_MAXK 16
#define MUSA_MIX_MAXQ 4
struct musa_mix_rows { const float * s[MUSA_MIX_MAXK]; int64_t ts[MUSA_MIX_MAXK]; };   // row j of token t: s[j] + t*ts[j]

// kernel 1: grid (k rows, n/256 segments); one element per thread; block sums of s^2 and s*q -> part[j][seg][1+nq]
#define MUSA_MIX_SEG 256
static __global__ void __launch_bounds__(MUSA_MIX_SEG) musa_mix_dots(const musa_mix_rows rows, const float * __restrict__ query,
        float * __restrict__ part, const int n, const int nq) {
    __shared__ float red[1 + MUSA_MIX_MAXQ][MUSA_MIX_SEG/16];
    const int j = blockIdx.x, seg = blockIdx.y, nseg = gridDim.y, t = threadIdx.x, tok = blockIdx.z;
    const int e = seg*MUSA_MIX_SEG + t;
    part += (int64_t) tok*gridDim.x*nseg*(1 + nq);
    float v = 0.0f, q[MUSA_MIX_MAXQ];
#pragma unroll
    for (int c = 0; c < MUSA_MIX_MAXQ; ++c) {
        q[c] = 0.0f;
    }
    if (e < n) {
        v = rows.s[j][(int64_t) tok*rows.ts[j] + e];
#pragma unroll
        for (int c = 0; c < MUSA_MIX_MAXQ; ++c) {
            if (c < nq) {
                q[c] = query[(int64_t) c*n + e];
            }
        }
    }
    // 16-thread groups: serial sums through shared memory (two levels, two barriers)
    __shared__ float vals[1 + MUSA_MIX_MAXQ][MUSA_MIX_SEG];
    vals[0][t] = v*v;
#pragma unroll
    for (int c = 0; c < MUSA_MIX_MAXQ; ++c) {
        if (c < nq) {
            vals[1 + c][t] = v*q[c];
        }
    }
    __syncthreads();
    if (t < (1 + nq)*(MUSA_MIX_SEG/16)) {
        const int c = t / (MUSA_MIX_SEG/16), g = t % (MUSA_MIX_SEG/16);
        float a = 0.0f;
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            a += vals[c][g*16 + u];
        }
        red[c][g] = a;
    }
    __syncthreads();
    if (t < 1 + nq) {
        float a = 0.0f;
#pragma unroll
        for (int g = 0; g < MUSA_MIX_SEG/16; ++g) {
            a += red[t][g];
        }
        part[((int64_t) j*nseg + seg)*(1 + nq) + t] = a;
    }
}

// kernel 2: grid (n/256 segments, nq); the block combines the partials and the k-way softmax once, then one output per thread
static __global__ void __launch_bounds__(MUSA_MIX_SEG) musa_mix_out(const musa_mix_rows rows, const float * __restrict__ part, float * __restrict__ out,
        const int n, const int k, const int nq, const int nseg, const float eps, const float scale) {
    __shared__ float inv_s[MUSA_MIX_MAXK], p_s[MUSA_MIX_MAXK], rs_s;
    const int q = blockIdx.y, t = threadIdx.x, tok = blockIdx.z;
    part += (int64_t) tok*k*nseg*(1 + nq);
    out  += (int64_t) tok*nq*n;
    const int e = blockIdx.x*MUSA_MIX_SEG + t;
    float sv[MUSA_MIX_MAXK];                        // all row loads in flight first
#pragma unroll
    for (int j = 0; j < MUSA_MIX_MAXK; ++j) {
        sv[j] = (j < k && e < n) ? rows.s[j][(int64_t) tok*rows.ts[j] + e] : 0.0f;
    }
    if (t < k) {
        float ss = 0.0f, dq = 0.0f;
        for (int g = 0; g < nseg; ++g) {
            ss += part[((int64_t) t*nseg + g)*(1 + nq)];
            dq += part[((int64_t) t*nseg + g)*(1 + nq) + 1 + q];
        }
        const float inv = 1.0f/sqrtf(ss/n + eps);
        inv_s[t] = inv;
        p_s[t] = scale*(dq*inv);
    }
    __syncthreads();
    if (t == 0) {
        float mx = -INFINITY;
        for (int j = 0; j < k; ++j) {
            mx = fmaxf(mx, p_s[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            const float x = expf(p_s[j] - mx);
            p_s[j] = x;
            sum += x;
        }
        rs_s = 1.0f/sum;
    }
    __syncthreads();
    if (e >= n) {
        return;
    }
    const float rs = rs_s;
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < MUSA_MIX_MAXK; ++j) {
        if (j < k) {
            acc = fmaf(sv[j]*inv_s[j], p_s[j]*rs, acc);
        }
    }
    out[(int64_t) q*n + e] = acc;
}

static bool musa_is_f32_row(const ggml_tensor * t, int64_t n) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->ne[0] == n && ggml_nrows(t) == 1;
}

static int musa_fuse_mix(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = getenv("GGML_MUSA_FUSE_MIX") == nullptr || atoi(getenv("GGML_MUSA_FUSE_MIX")) != 0;
    if (!on) {
        return 0;
    }
    int idx[MUSA_MIX_MAXK + 8];
    ggml_op ops[MUSA_MIX_MAXK + 8];
    int cnt = 0;
    const ggml_tensor * concats[MUSA_MIX_MAXK];
    int nc = 0;
    int j = i;
    // the concat chain: each next concat takes the previous one as src1
    while (j < cgraph->n_nodes && nc < MUSA_MIX_MAXK - 1) {
        const ggml_tensor * c = cgraph->nodes[j];
        if (c->op != GGML_OP_CONCAT || ggml_get_op_params_i32(c, 0) != 1 || (nc > 0 && c->src[1] != concats[nc - 1])) {
            break;
        }
        concats[nc++] = c;
        idx[cnt] = j; ops[cnt++] = GGML_OP_CONCAT;
        ++j;
    }
    if (nc == 0 || j + 6 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * states = concats[nc - 1];
    ggml_tensor * const * nd = cgraph->nodes + j;
    // RMS_NORM, TRANSPOSE, CONT, REPEAT, MUL_MAT, SOFT_MAX, MUL_MAT
    const ggml_op want[7] = { GGML_OP_RMS_NORM, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_REPEAT, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT };
    for (int u = 0; u < 7; ++u) {
        if (nd[u]->op != want[u]) {
            return 0;
        }
        idx[cnt] = j + u; ops[cnt++] = want[u];
    }
    const ggml_tensor * rms = nd[0], * tr = nd[1], * ct = nd[2], * rep = nd[3], * mm1 = nd[4], * sm = nd[5];
    ggml_tensor * out = nd[6];
    if (rms->src[0] != states || tr->src[0] != rms || ct->src[0] != tr || mm1->src[0] != rms || mm1->src[1] != rep ||
        sm->src[0] != mm1 || sm->src[1] != nullptr || out->src[0] != ct || out->src[1] != sm) {
        return 0;
    }
    const int64_t n = states->ne[0], k = states->ne[1], nq = rep->ne[1];
    const ggml_tensor * query = rep->src[0];
    float scale, max_bias;
    memcpy(&scale, (const float *) sm->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) sm->op_params + 1, sizeof(float));
    const int64_t Nt = states->ne[2];
    if (Nt < 1 || Nt > 65535 || states->ne[3] != 1 || k != nc + 1 || k > MUSA_MIX_MAXK || nq > MUSA_MIX_MAXQ || nq < 1 ||
        max_bias != 0.0f || query->type != GGML_TYPE_F32 || !ggml_is_contiguous(query) || query->ne[0] != n || query->ne[1] != nq ||
        ggml_nelements(query) != n*nq || rep->ne[0] != n || ggml_nelements(rep) != n*nq*Nt ||
        out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || out->ne[0] != n || ggml_nelements(out) != n*nq*Nt ||
        rms->type != GGML_TYPE_F32 || !ggml_is_contiguous(rms) || n > INT32_MAX) {
        return 0;
    }
    if (!ggml_can_fuse_subgraph_ext(cgraph, idx, cnt, ops, &idx[cnt - 1], 1)) {
        return 0;
    }
    // rows in the order of the stacked states: [last concat's src0, ..., first concat's src0, first concat's src1]
    musa_mix_rows rows{};
    // each state row: [n, 1, Nt] (token stride nb2) - one token: a single row
    auto row_ok = [&](const ggml_tensor * t, int64_t & ts) {
        if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) || t->ne[0] != n || t->ne[1] != 1 || t->ne[2] != Nt || t->ne[3] != 1) {
            return false;
        }
        ts = (int64_t) (t->nb[2]/sizeof(float));
        return true;
    };
    for (int c = nc - 1, r = 0; c >= 0; --c, ++r) {
        if (!row_ok(concats[c]->src[0], rows.ts[r])) {
            return 0;
        }
        rows.s[r] = (const float *) concats[c]->src[0]->data;
    }
    if (!row_ok(concats[0]->src[1], rows.ts[nc])) {
        return 0;
    }
    rows.s[nc] = (const float *) concats[0]->src[1]->data;
    bool out_direct = musa_disjoint_impl({ out }, { query }, __LINE__);
    for (int r = 0; r <= nc && out_direct; ++r) {
        const ggml_tensor * row = r < nc ? concats[r]->src[0] : concats[0]->src[1];
        out_direct = musa_disjoint_impl({ out }, { row }, __LINE__);
    }
    const float eps = ggml_get_op_params_f32(rms, 0);
    static const bool dbg = getenv("GGML_MUSA_FUSE_DBG") != nullptr;
    static int n_fused = 0;
    if (dbg && ++n_fused % 1000 == 1) {
        fprintf(stderr, "musa_fuse_mix: %d fused (k=%d nq=%d)\n", n_fused, (int) k, (int) nq);
    }
    if (musa_skip_fused("mix")) {
        return idx[cnt - 1] - i;
    }
    cudaStream_t st = ctx.stream();
    // own scratch for the k x (1+nq) sums (the RMS node's memory may share space with a state row); the output goes
    // through a temporary when the allocator placed it over an input
    const int nseg = (int) ((n + MUSA_MIX_SEG - 1)/MUSA_MIX_SEG);
    ggml_cuda_pool_alloc<float> raw(ctx.pool(), (size_t) k*nseg*(1 + nq)*Nt);
    ggml_cuda_pool_alloc<float> tmp(ctx.pool());
    float * od = out_direct ? (float *) out->data : tmp.alloc((size_t) n*nq*Nt);
    musa_mix_dots<<<dim3((unsigned) k, (unsigned) nseg, (unsigned) Nt), MUSA_MIX_SEG, 0, st>>>(rows, (const float *) query->data, raw.get(), (int) n, (int) nq);
    musa_mix_out<<<dim3((unsigned) nseg, (unsigned) nq, (unsigned) Nt), MUSA_MIX_SEG, 0, st>>>(rows, raw.get(), od,
        (int) n, (int) k, (int) nq, nseg, eps, scale);
    if (!out_direct) {
        musa_copy_small(st, out->data, od, (size_t) n*nq*Nt*sizeof(float));
    }
    return idx[cnt - 1] - i;
}

// ---- Inkling short convolution without input-dependent taps, convolve(), one token:
//   CONCAT(past [L-1, C], x^T [1, C]) dim 0, VIEW, VIEW, CPY (state store), SSM_CONV, RESHAPE, ADD(x, conv), SCALE -> 1 kernel
//   per channel c: w = [past(0..L-2, c), x_c]; state(c) = w[1..L-1]; out_c = (x_c + sum_t w_t*k(t, c))*s + b
#define MUSA_CONV_MAXT 8
template <int L>
static __global__ void musa_short_conv(const float * __restrict__ past, const int64_t past_nb1, const float * __restrict__ x,
        const float * __restrict__ k, float * __restrict__ state, float * __restrict__ out, const int C, const float s, const float b,
        const float * __restrict__ res, const int T) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= C) {
        return;
    }
    float w[L - 1 + MUSA_CONV_MAXT];
#pragma unroll
    for (int t = 0; t < L - 1; ++t) {
        w[t] = past[(int64_t) c*past_nb1 + t];
    }
#pragma unroll
    for (int t = 0; t < MUSA_CONV_MAXT; ++t) {
        if (t < T) {
            w[L - 1 + t] = x[(int64_t) t*C + c];
        }
    }
    float kk[L];
#pragma unroll
    for (int j = 0; j < L; ++j) {
        kk[j] = k[c*L + j];
    }
#pragma unroll
    for (int t = 0; t < MUSA_CONV_MAXT; ++t) {
        if (t < T) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < L; ++j) {
                acc += w[t + j]*kk[j];
            }
            const float y = (w[L - 1 + t] + acc)*s + b;
            out[(int64_t) t*C + c] = res ? res[(int64_t) t*C + c] + y : y;     // optional residual ADD(res, y)
        }
    }
    // the state keeps the last L-1 inputs of the window
#pragma unroll
    for (int j = 0; j < L - 1; ++j) {
        float v = 0.0f;
#pragma unroll
        for (int t = 0; t < L - 1 + MUSA_CONV_MAXT; ++t) {
            if (t == T + j) {
                v = w[t];
            }
        }
        state[(int64_t) c*(L - 1) + j] = v;
    }
}

static int musa_fuse_short_conv(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = getenv("GGML_MUSA_FUSE_CONV") == nullptr || atoi(getenv("GGML_MUSA_FUSE_CONV")) != 0;
    if (!on || i + 7 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_op want[8] = { GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE };
    ggml_tensor * const * nd = cgraph->nodes + i;
    for (int u = 0; u < 8; ++u) {
        if (nd[u]->op != want[u]) {
            return 0;
        }
    }
    const ggml_tensor * cat = nd[0], * vsrc = nd[1], * vdst = nd[2], * cpy = nd[3], * conv = nd[4], * rs = nd[5], * add = nd[6];
    ggml_tensor * out = nd[7];
    const ggml_tensor * past = cat->src[0], * xt = cat->src[1], * kw = conv->src[1];
    const int64_t C = cat->ne[1], L = kw->ne[0], T = cat->ne[0] - (L - 1);     // window L, T new tokens
    // x^T is a transposed view of the (contiguous) projection output; add takes that projection as src0
    const ggml_tensor * x = add->src[0];
    if (ggml_get_op_params_i32(cat, 0) != 0 || L != 4 || T < 1 || T > MUSA_CONV_MAXT || cat->ne[2] != 1 || cat->ne[3] != 1 ||
        cat->type != GGML_TYPE_F32 || past->type != GGML_TYPE_F32 || past->ne[0] != L - 1 || past->ne[1] != C || past->nb[0] != sizeof(float) ||
        xt->ne[0] != T || xt->ne[1] != C || ggml_nelements(xt) != C*T ||
        vsrc->src[0] != cat || vsrc->view_src != cat || vsrc->view_offs != (size_t) T*sizeof(float) || vsrc->ne[0] != L - 1 || vsrc->ne[1] != C ||
        cpy->src[0] != vsrc || cpy->src[1] != vdst || !ggml_is_contiguous(vdst) || vdst->type != GGML_TYPE_F32 || ggml_nelements(vdst) != (L - 1)*C ||
        conv->src[0] != cat || kw->type != GGML_TYPE_F32 || !ggml_is_contiguous(kw) || kw->ne[0] != L || kw->ne[1] != C ||
        rs->src[0] != conv || add->src[1] != rs || out->src[0] != add ||
        x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || ggml_nelements(x) != C*T || x->ne[0] != C || x->data != xt->data ||
        !ggml_is_contiguous(out) || ggml_nelements(out) != C*T || C > INT32_MAX) {
        return 0;
    }
    // optional residual: the next non-view node is ADD(res, out) and the convolution result has no other reader
    static const bool res_on = getenv("GGML_MUSA_FUSE_CONV_RES") == nullptr || atoi(getenv("GGML_MUSA_FUSE_CONV_RES")) != 0;
    int ja = i + 8;
    while (res_on && ja < cgraph->n_nodes && ja < i + 12 &&
           (cgraph->nodes[ja]->op == GGML_OP_RESHAPE || cgraph->nodes[ja]->op == GGML_OP_VIEW)) {
        ++ja;
    }
    ggml_tensor * radd = nullptr;
    const ggml_tensor * res = nullptr;
    if (res_on && ja < cgraph->n_nodes && cgraph->nodes[ja]->op == GGML_OP_ADD && cgraph->nodes[ja]->src[1] == out &&
        ggml_node_get_use_count(cgraph, i + 7) == 1) {
        radd = cgraph->nodes[ja];
        res = radd->src[0];
        if (res->type != GGML_TYPE_F32 || !ggml_is_contiguous(res) || ggml_nelements(res) != C*T ||
            !ggml_is_contiguous(radd) || ggml_nelements(radd) != C*T || radd->type != GGML_TYPE_F32) {
            radd = nullptr;
            res = nullptr;
        }
    }
    // concat / conv / add results are internal; the CPY into the state and the SCALE (or residual ADD) result are the outputs
    // (the VIEW of the persistent state buffer, i + 2, is left out: the generic check rejects views of non-constant
    // external tensors; it is only the CPY destination)
    const int idx[8] = { i, i + 1, i + 3, i + 4, i + 5, i + 6, i + 7, ja };
    const ggml_op ops8[8] = { GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE, GGML_OP_ADD };
    const int outs[2] = { i + 3, radd ? ja : i + 7 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idx, radd ? 8 : 7, ops8, outs, 2)) {
        return 0;
    }
    ggml_tensor * dst = radd ? radd : out;
    // dst may be x or res itself (in-place): each thread reads x[c] / res[c] before writing dst[c]
    if (!musa_disjoint_impl({ vdst }, { past, x, kw, res }, __LINE__) || !musa_disjoint_impl({ dst }, { past, kw }, __LINE__) ||
        (dst->data != x->data && !musa_disjoint_impl({ dst }, { x }, __LINE__)) ||
        (res && dst->data != res->data && !musa_disjoint_impl({ dst }, { res }, __LINE__)) ||
        musa_overlap(vdst->data, ggml_nbytes(vdst), dst->data, ggml_nbytes(dst))) {
        return 0;
    }
    if (musa_skip_fused("conv")) {
        return radd ? ja - i : 7;
    }
    const float s = ggml_get_op_params_f32(out, 0), b = ggml_get_op_params_f32(out, 1);
    musa_short_conv<4><<<(unsigned) ((C + 255)/256), 256, 0, ctx.stream()>>>((const float *) past->data, past->nb[1]/sizeof(float),
        (const float *) x->data, (const float *) kw->data, (float *) vdst->data, (float *) dst->data, (int) C, s, b,
        res ? (const float *) res->data : nullptr, (int) T);
    return radd ? ja - i : 7;
}

// ---- Inkling SITU expert activation on the gate/up halves of the fused gate_up product:
//   out = (sg1*tanh(sg0*g + bg0) + bg1) * sigmoid(g) * (su1*tanh(su0*u + bu0) + bu1)
//   SCALE(g), TANH, SCALE, SIGMOID(g), MUL, [VIEW u], SCALE(u), TANH, SCALE, MUL  ->  1 kernel
static __global__ void musa_situ(const float * __restrict__ g, const float * __restrict__ u, const int64_t g_nb1, const int64_t u_nb1,
        float * __restrict__ out, const int ne0, const int nrows,
        const float sg0, const float bg0, const float sg1, const float bg1, const float su0, const float bu0, const float su1, const float bu1) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t r = idx / ne0, c = idx - r*ne0;    // one 64-bit division per element: fine for ~10k elements
    if (r >= nrows) {
        return;
    }
    const float gv = g[r*g_nb1 + c], uv = u[r*u_nb1 + c];
    const float tg = tanhf(gv*sg0 + bg0)*sg1 + bg1;
    const float sig = 1.0f/(1.0f + expf(-gv));
    const float tu = tanhf(uv*su0 + bu0)*su1 + bu1;
    out[idx] = (tg*sig)*tu;
}

static int musa_fuse_situ(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = getenv("GGML_MUSA_FUSE_SITU") == nullptr || atoi(getenv("GGML_MUSA_FUSE_SITU")) != 0;
    if (!on || i + 9 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_op want[10] = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL,
                               GGML_OP_VIEW, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL };
    ggml_tensor * const * nd = cgraph->nodes + i;
    for (int u = 0; u < 10; ++u) {
        if (nd[u]->op != want[u]) {
            return 0;
        }
    }
    const ggml_tensor * g = nd[0]->src[0], * uview = nd[5];
    ggml_tensor * out = nd[9];
    if (ggml_get_unary_op(nd[1]) != GGML_UNARY_OP_TANH || ggml_get_unary_op(nd[3]) != GGML_UNARY_OP_SIGMOID ||
        ggml_get_unary_op(nd[7]) != GGML_UNARY_OP_TANH ||
        nd[1]->src[0] != nd[0] || nd[2]->src[0] != nd[1] || nd[3]->src[0] != g || nd[4]->src[0] != nd[2] || nd[4]->src[1] != nd[3] ||
        nd[6]->src[0] != uview || nd[7]->src[0] != nd[6] || nd[8]->src[0] != nd[7] || out->src[0] != nd[4] || out->src[1] != nd[8]) {
        return 0;
    }
    const int64_t ne0 = g->ne[0], nrows = ggml_nrows(g);
    if (g->type != GGML_TYPE_F32 || uview->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
        g->nb[0] != sizeof(float) || uview->nb[0] != sizeof(float) || !ggml_are_same_shape(g, uview) || !ggml_are_same_shape(g, out) ||
        g->ne[3] != 1 || g->nb[2] != g->nb[1]*g->ne[1] || uview->nb[2] != uview->nb[1]*uview->ne[1] ||   // rows evenly strided over dim 2
        !ggml_is_contiguous(out) || ne0*nrows > INT32_MAX) {
        return 0;
    }
    // the u VIEW (i + 5) views the external gate_up product: left out of the generic check, used only by node i + 6
    const int idx[9] = { i, i + 1, i + 2, i + 3, i + 4, i + 6, i + 7, i + 8, i + 9 };
    const ggml_op ops9[9] = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL,
                              GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL };
    const int outs[1] = { i + 9 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idx, 9, ops9, outs, 1)) {
        return 0;
    }
    if (!musa_disjoint_impl({ out }, { g, uview }, __LINE__)) {
        return 0;
    }
    if (musa_skip_fused("situ")) {
        return 9;
    }
    auto P = [](const ggml_tensor * t, int k) { return ggml_get_op_params_f32(t, k); };
    const int64_t n = ne0*nrows;
    musa_situ<<<(unsigned) ((n + 255)/256), 256, 0, ctx.stream()>>>((const float *) g->data, (const float *) uview->data,
        g->nb[1]/sizeof(float), uview->nb[1]/sizeof(float), (float *) out->data, (int) ne0, (int) nrows,
        P(nd[0], 0), P(nd[0], 1), P(nd[2], 0), P(nd[2], 1), P(nd[6], 0), P(nd[6], 1), P(nd[8], 0), P(nd[8], 1));
    return 9;
}

// ---- Inkling relative-position bias + mask for FlashAttention, one token:
//   MUL_MAT(table16 [nr, T], R [nr, H]), [RESHAPE sink], SUB, PERMUTE, CONT, GET_ROWS(rel_pos), RESHAPE, PERMUTE, CONT,
//   CPY(mask copy), ADD, CPY(-> f16)   ->  1 kernel:   out[h][p] = (R_h . table16[:, rel_pos[p]] - sink_h) + mask[p]
template <int NR>
static __global__ void musa_rel_bias(const float * __restrict__ table, const float * __restrict__ R, const float * __restrict__ sink,
        const int32_t * __restrict__ pos, const half * __restrict__ mask, float * __restrict__ mask_f32, half * __restrict__ out, const int n_kv,
        const int64_t out_nb_h, const int T0, const int H, const int64_t pos_nb1, const int64_t mask_nb1, const int64_t out_nb_t) {
    const int h = blockIdx.y, t = blockIdx.z;          // head, token of the batch
    const int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= n_kv) {
        return;
    }
    const int d = pos[(int64_t) t*pos_nb1 + p];
    const float * tr = table + (int64_t) d*NR;
    const float * r = R + ((int64_t) t*H + h)*NR;
    float acc = 0.0f;
    if (d < T0) {                 // rows past T0 are the zero padding of the table
#pragma unroll
        for (int k = 0; k < NR; ++k) {
            acc = fmaf(r[k], tr[k], acc);
        }
    }
    const float mv = __half2float(mask[(int64_t) t*mask_nb1 + p]);
    if (h == 0) {
        mask_f32[(int64_t) t*n_kv + p] = mv;     // the f32 mask copy of the graph (it may have readers outside this chain)
    }
    out[(int64_t) h*out_nb_h + (int64_t) t*out_nb_t + p] = __float2half((acc - sink[(int64_t) t*H + h]) + mv);
}

static int musa_fuse_rel_bias(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, const ggml_tensor * pad = nullptr) {
    static const bool on = getenv("GGML_MUSA_FUSE_RELB") == nullptr || atoi(getenv("GGML_MUSA_FUSE_RELB")) != 0;
    if (!on || i + 11 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_op want[12] = { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_SUB, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_GET_ROWS,
                               GGML_OP_RESHAPE, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_CPY, GGML_OP_ADD, GGML_OP_CPY };
    ggml_tensor * const * nd = cgraph->nodes + i;
    for (int u = 0; u < 12; ++u) {
        if (nd[u]->op != want[u]) {
            return 0;
        }
    }
    const ggml_tensor * mm = nd[0], * srs = nd[1], * sub = nd[2], * pm1 = nd[3], * c1 = nd[4], * gr = nd[5];
    const ggml_tensor * rs2 = nd[6], * pm2 = nd[7], * kqb = nd[8], * mcp = nd[9], * add = nd[10];
    ggml_tensor * out = nd[11];
    const ggml_tensor * table = mm->src[0], * R = mm->src[1], * sink = srs->src[0], * pos = gr->src[1], * mask = mcp->src[0];
#define RELB_REQ_PRE(c) if (c) { return 0; }
    const int64_t NR = table->ne[0], T = table->ne[1], H = R->ne[1], n_kv = pos->ne[0], Nt = R->ne[2];
    // with the PAD folded in, the kernel reads the unpadded table (rows >= T0 are zero)
    const ggml_tensor * rtab = pad ? pad->src[0] : table;
    const int64_t T0 = pad ? rtab->ne[1] : T;
    RELB_REQ_PRE(pad && (table != pad || rtab->type != GGML_TYPE_F32 || !ggml_is_contiguous(rtab) || rtab->ne[0] != NR || T0 > T))
    static const bool rdbg = getenv("GGML_MUSA_FUSE_DBG") != nullptr;
#define RELB_REQ(c) if (c) { if (rdbg) { static int np = 0; if (np++ < 2) fprintf(stderr, "rel_bias: fails %s\n", #c); } return 0; }
    RELB_REQ(NR != 16)
    RELB_REQ(R->ne[0] != NR)
    RELB_REQ(ggml_nelements(R) != NR*H*Nt || Nt > 65535)
    RELB_REQ(table->type != GGML_TYPE_F32)
    RELB_REQ(R->type != GGML_TYPE_F32)
    RELB_REQ(!ggml_is_contiguous(table))
    RELB_REQ(!ggml_is_contiguous(R))
    RELB_REQ(sub->src[0] != mm)
    RELB_REQ(sub->src[1] != srs)
    RELB_REQ(sink->type != GGML_TYPE_F32)
    RELB_REQ(!ggml_is_contiguous(sink))
    RELB_REQ(ggml_nelements(sink) != H*Nt)
    RELB_REQ(pm1->src[0] != sub)
    RELB_REQ(c1->src[0] != pm1)
    RELB_REQ(gr->src[0] != c1)
    RELB_REQ(c1->ne[0] != H)
    RELB_REQ(c1->ne[1] != T)
    RELB_REQ(pos->type != GGML_TYPE_I32)
    RELB_REQ(pos->nb[0] != sizeof(int32_t))
    RELB_REQ(ggml_nelements(pos) != n_kv*Nt)
    RELB_REQ(rs2->src[0] != gr)
    RELB_REQ(pm2->src[0] != rs2)
    RELB_REQ(kqb->src[0] != pm2)
    RELB_REQ(kqb->ne[0] != n_kv)
    RELB_REQ(kqb->ne[1] != Nt)
    RELB_REQ(kqb->ne[2] != H)
    RELB_REQ(kqb->ne[3] != 1)
    RELB_REQ(mask->type != GGML_TYPE_F16)
    RELB_REQ(mask->ne[0] != n_kv)
    RELB_REQ(mask->nb[0] != sizeof(half) || mask->ne[1] < Nt || mask->ne[2] != 1 || mask->ne[3] != 1)
    RELB_REQ(mcp->type != GGML_TYPE_F32)
    RELB_REQ(add->src[0] != kqb)
    RELB_REQ(add->src[1] != mcp)
    RELB_REQ(out->src[0] != add)
    RELB_REQ(out->type != GGML_TYPE_F16)
    RELB_REQ(out->nb[0] != sizeof(half))
    RELB_REQ(!ggml_are_same_shape(out, kqb))
    RELB_REQ(n_kv > INT32_MAX)
    // the sink RESHAPE (i + 1) views an external matmul result: left out of the generic check (used only by the SUB)
    // the mask CPY (i + 9) is a view of its own external destination: left out of the generic check; the kernel
    // writes its f32 result too, so readers outside the chain still see it
    const int idx[10] = { i, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7, i + 8, i + 10, i + 11 };
    const ggml_op ops10[10] = { GGML_OP_MUL_MAT, GGML_OP_SUB, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_GET_ROWS,
                                GGML_OP_RESHAPE, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_ADD, GGML_OP_CPY };
    const int outs[1] = { i + 11 };
    RELB_REQ(!ggml_is_contiguous(mcp) || ggml_nelements(mcp) != n_kv*Nt)
    if (pad) {
        const int idx11[11] = { i - 1, i, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7, i + 8, i + 10, i + 11 };
        const ggml_op ops11[11] = { GGML_OP_PAD, GGML_OP_MUL_MAT, GGML_OP_SUB, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_GET_ROWS,
                                    GGML_OP_RESHAPE, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_ADD, GGML_OP_CPY };
        RELB_REQ(!ggml_can_fuse_subgraph_ext(cgraph, idx11, 11, ops11, outs, 1))
    } else {
        RELB_REQ(!ggml_can_fuse_subgraph_ext(cgraph, idx, 10, ops10, outs, 1))
    }
#undef RELB_REQ
#undef RELB_REQ_PRE
    // outputs are written at the chain's first node; if the allocator placed them over still-needed inputs (R, sink:
    // dead by the original position of the outputs) the f16 result goes to a temporary buffer and is copied after
    if (!ggml_is_contiguous(out) || musa_overlap(out->data, ggml_nbytes(out), mcp->data, ggml_nbytes(mcp))) {
        return 0;
    }
    const bool out_direct = musa_disjoint_impl({ out }, { rtab, R, sink, pos, mask }, __LINE__);
    const bool mcp_direct = musa_disjoint_impl({ mcp }, { rtab, R, sink, pos, mask }, __LINE__);
    static const bool dbg = getenv("GGML_MUSA_FUSE_DBG") != nullptr;
    static int n_fused = 0;
    if (dbg && ++n_fused % 1000 == 1) {
        fprintf(stderr, "musa_fuse_rel_bias: %d fused (n_kv=%d)\n", n_fused, (int) n_kv);
    }
    if (musa_skip_fused("relb")) {
        return 11;
    }
    ggml_cuda_pool_alloc<half> tmp(ctx.pool());
    ggml_cuda_pool_alloc<float> tmpm(ctx.pool());
    half * od = out_direct ? (half *) out->data : tmp.alloc((size_t) H*n_kv*Nt);
    float * md = mcp_direct ? (float *) mcp->data : tmpm.alloc((size_t) n_kv*Nt);
    const int64_t od_nb_h = out_direct ? (int64_t) (out->nb[2]/sizeof(half)) : n_kv*Nt;
    const int64_t od_nb_t = out_direct ? (int64_t) (out->nb[1]/sizeof(half)) : n_kv;
    musa_rel_bias<16><<<dim3((unsigned) ((n_kv + 127)/128), (unsigned) H, (unsigned) Nt), 128, 0, ctx.stream()>>>((const float *) rtab->data,
        (const float *) R->data, (const float *) sink->data, (const int32_t *) pos->data, (const half *) mask->data, md,
        od, (int) n_kv, od_nb_h, (int) T0, (int) H, pos->nb[1]/sizeof(int32_t), mask->nb[1]/sizeof(half), od_nb_t);
    if (!out_direct) {
        musa_copy_small(ctx.stream(), out->data, od, ggml_nbytes(out));
    }
    if (!mcp_direct) {
        musa_copy_small(ctx.stream(), mcp->data, md, (size_t) n_kv*Nt*sizeof(float));
    }
    return 11;
}

// ---- Inkling relative-position bias table for the in-attention bias (ggml_flash_attn_ext_set_rel):
//   [PAD] MUL_MAT(table16 [nr, T], R [nr, H, Nt]), RESHAPE sink, SUB  ->  1 kernel:
//   out[t][h][d] = (d < T0 ? R[t,h] . table16[:, d] : 0) - sink[t, h]
template <int NR>
static __global__ void musa_rel_table(const float * __restrict__ table, const float * __restrict__ R, const float * __restrict__ sink,
        float * __restrict__ out, const int T, const int T0, const int H) {
    const int h = blockIdx.y, t = blockIdx.z;
    const int d = blockIdx.x*blockDim.x + threadIdx.x;
    if (d >= T) {
        return;
    }
    const float * r = R + ((int64_t) t*H + h)*NR;
    float acc = 0.0f;
    if (d < T0) {
        const float * tr = table + (int64_t) d*NR;
#pragma unroll
        for (int k = 0; k < NR; ++k) {
            acc = fmaf(r[k], tr[k], acc);
        }
    }
    out[((int64_t) t*H + h)*T + d] = acc - sink[(int64_t) t*H + h];
}

static int musa_fuse_rel_table(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, const ggml_tensor * pad = nullptr) {
    static const bool on = getenv("GGML_MUSA_FUSE_RELT") == nullptr || atoi(getenv("GGML_MUSA_FUSE_RELT")) != 0;
    if (!on || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * const * nd = cgraph->nodes + i;
    if (nd[0]->op != GGML_OP_MUL_MAT || nd[1]->op != GGML_OP_RESHAPE || nd[2]->op != GGML_OP_SUB) {
        return 0;
    }
    const ggml_tensor * mm = nd[0], * srs = nd[1];
    ggml_tensor * out = nd[2];
    const ggml_tensor * table = mm->src[0], * R = mm->src[1], * sink = srs->src[0];
    const ggml_tensor * rtab = pad ? pad->src[0] : table;
    const int64_t NR = table->ne[0], T = table->ne[1], H = R->ne[1], Nt = R->ne[2];
    const int64_t T0 = pad ? rtab->ne[1] : T;
    if (out->src[0] != mm || out->src[1] != srs || NR != 16 || R->ne[0] != NR || ggml_nelements(R) != NR*H*Nt || Nt > 65535 ||
        (pad && (table != pad || rtab->ne[0] != NR || T0 > T)) || rtab->type != GGML_TYPE_F32 || !ggml_is_contiguous(rtab) ||
        table->type != GGML_TYPE_F32 || R->type != GGML_TYPE_F32 || !ggml_is_contiguous(R) ||
        sink->type != GGML_TYPE_F32 || !ggml_is_contiguous(sink) || ggml_nelements(sink) != H*Nt ||
        out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || out->ne[0] != T || out->ne[1] != H || out->ne[2] != Nt) {
        return 0;
    }
    const int outs[1] = { i + 2 };
    if (pad) {
        const int idx[3] = { i - 1, i, i + 2 };
        const ggml_op ops[3] = { GGML_OP_PAD, GGML_OP_MUL_MAT, GGML_OP_SUB };
        if (!ggml_can_fuse_subgraph_ext(cgraph, idx, 3, ops, outs, 1)) {
            return 0;
        }
    } else {
        const int idx[2] = { i, i + 2 };
        const ggml_op ops[2] = { GGML_OP_MUL_MAT, GGML_OP_SUB };
        if (!ggml_can_fuse_subgraph_ext(cgraph, idx, 2, ops, outs, 1)) {
            return 0;
        }
    }
    // the output is written at the chain's first node: it must not overlap the inputs still to be read
    if (!musa_disjoint_impl({ out }, { rtab, R, sink }, __LINE__)) {
        return 0;
    }
    if (musa_skip_fused("relt")) {
        return 2;
    }
    musa_rel_table<16><<<dim3((unsigned) ((T + 127)/128), (unsigned) H, (unsigned) Nt), 128, 0, ctx.stream()>>>(
        (const float *) rtab->data, (const float *) R->data, (const float *) sink->data, (float *) out->data, (int) T, (int) T0, (int) H);
    return 2;
}

// controller including its input norms: [CONT x,] RMS_NORM x, VIEW m, CONT m, RMS_NORM m, then the 10 nodes above
// (x / m are single contiguous rows read in place; the CONT copies and normalized rows are never materialized)
static int musa_fuse_control_norm(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = getenv("GGML_MUSA_FUSE_CTRL") == nullptr || atoi(getenv("GGML_MUSA_FUSE_CTRL")) != 0;
    static const bool on2 = getenv("GGML_MUSA_FUSE_CTRL_NORM") == nullptr || atoi(getenv("GGML_MUSA_FUSE_CTRL_NORM")) != 0;
    if (!on || !on2) {
        return 0;
    }
    const bool has_cx = cgraph->nodes[i]->op == GGML_OP_CONT;
    const int j = has_cx ? i + 1 : i;     // RMS_NORM x
    if (j + 13 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * const * nd = cgraph->nodes + j;
    const ggml_op want[14] = { GGML_OP_RMS_NORM, GGML_OP_VIEW, GGML_OP_CONT, GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_MUL_MAT,
                               GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD };
    for (int u = 0; u < 14; ++u) {
        if (nd[u]->op != want[u]) {
            return 0;
        }
    }
    const ggml_tensor * rx = nd[0], * vm = nd[1], * cm = nd[2], * rm = nd[3], * cat = nd[4], * down = nd[5], * mmf = nd[6], * tf = nd[7];
    const ggml_tensor * mf = nd[8], * sb = nd[9], * mmg = nd[10], * tg = nd[11], * mg = nd[12];
    ggml_tensor * out = nd[13];
    const ggml_tensor * xn = has_cx ? cgraph->nodes[i] : rx->src[0];     // the x node the graph uses (CONT or view)
    const ggml_tensor * xsrc = has_cx ? cgraph->nodes[i]->src[0] : rx->src[0];
    const ggml_tensor * msrc = vm;
    if (ggml_get_unary_op(tf) != GGML_UNARY_OP_TANH || ggml_get_unary_op(tg) != GGML_UNARY_OP_TANH ||
        rx->src[0] != xn || cm->src[0] != vm || rm->src[0] != cm || cat->src[0] != rx || cat->src[1] != rm ||
        ggml_get_op_params_i32(cat, 0) != 0 || down->src[1] != cat || mmf->src[1] != down || mmg->src[1] != down ||
        tf->src[0] != mmf || mf->src[0] != tf || mf->src[1] != xn || sb->src[0] != xn || sb->src[1] != mf ||
        tg->src[0] != mmg || mg->src[0] != tg || mg->src[1] != cm || out->src[0] != sb || out->src[1] != mg) {
        return 0;
    }
    const ggml_tensor * wd = down->src[0], * wf = mmf->src[0], * wg = mmg->src[0];
    const int64_t n = xn->ne[0], rank = wd->ne[1], Nt = xn->ne[1];     // Nt tokens (MTP verify batches)
    if (wd->type != GGML_TYPE_Q8_0 || wf->type != GGML_TYPE_Q8_0 || wg->type != GGML_TYPE_Q8_0 ||
        !ggml_is_contiguous(wd) || !ggml_is_contiguous(wf) || !ggml_is_contiguous(wg) ||
        rank != 64 || wd->ne[0] != 2*n || n % 32 != 0 || wf->ne[0] != rank || wg->ne[0] != rank || wf->ne[1] != n || wg->ne[1] != n ||
        xsrc->type != GGML_TYPE_F32 || xsrc->nb[0] != sizeof(float) || xsrc->ne[0] != n || xsrc->ne[1] != Nt || ggml_nrows(xsrc) != Nt ||
        msrc->type != GGML_TYPE_F32 || msrc->nb[0] != sizeof(float) || msrc->ne[0] != n || msrc->ne[1] != Nt || ggml_nrows(msrc) != Nt ||
        !musa_is_f32_vec(down, rank*Nt) || !musa_is_f32_vec(out, n*Nt) || Nt > 65535 ||
        ggml_nelements(xn) != n*Nt || ggml_nelements(cm) != n*Nt || n > INT32_MAX ||
        ggml_cuda_q4_0_mma_is_tiled(wd) || ggml_cuda_q4_0_mma_is_tiled(wf) || ggml_cuda_q4_0_mma_is_tiled(wg)) {
        return 0;
    }
    // generic check over the computed nodes (the VIEW of the external messages, j + 1, is left out: used only by CONT m)
    int idx[15];
    ggml_op ops[15];
    int c = 0;
    if (has_cx) {
        idx[c] = i; ops[c++] = GGML_OP_CONT;
    }
    for (int u = 0; u < 14; ++u) {
        if (u == 1) {
            continue;
        }
        idx[c] = j + u; ops[c++] = want[u];
    }
    const int outs[1] = { j + 13 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idx, c, ops, outs, 1)) {
        return 0;
    }
    if (!musa_disjoint_impl({ down, out }, { xsrc, msrc, wd, wf, wg }, __LINE__) || musa_overlap(down->data, ggml_nbytes(down), out->data, ggml_nbytes(out))) {
        return 0;
    }
    if (musa_skip_fused("ctrl")) {
        return (j + 13) - i;
    }
    cudaStream_t st = ctx.stream();
    const int64_t xts = Nt > 1 ? (int64_t) (xsrc->nb[1]/sizeof(float)) : 0, mts = Nt > 1 ? (int64_t) (msrc->nb[1]/sizeof(float)) : 0;
    musa_ctrl_down<true><<<dim3((unsigned) ((rank + 3)/4), (unsigned) Nt), 512, 0, st>>>((const block_q8_0 *) wd->data, (const float *) xsrc->data,
        (const float *) msrc->data, (float *) down->data, (int) n, (int) rank, ggml_get_op_params_f32(rx, 0), ggml_get_op_params_f32(rm, 0), xts, mts);
    musa_ctrl_up<64><<<dim3((unsigned) ((n + 63)/64), (unsigned) Nt), 256, 0, st>>>((const block_q8_0 *) wf->data, (const block_q8_0 *) wg->data,
        (const float *) down->data, (const float *) xsrc->data, (const float *) msrc->data, (float *) out->data, (int) n, xts, mts);
    return (j + 13) - i;
}

// ---- MoE router post-processing, sigmoid gating with selection bias and normalized weights, one token:
//   SIGMOID, RESHAPE, ADD(bias), ARGSORT, VIEW(top k), GET_ROWS, RESHAPE, SUM_ROWS, SCALE(eps), DIV, RESHAPE, SCALE(w)  ->  1 kernel
//   each thread = one expert; its rank among the biased scores (ties: lower index first) selects it into the top k
template <int NE>
static __global__ void __launch_bounds__(NE) musa_router(const float * __restrict__ logits, const float * __restrict__ bias,
        int32_t * __restrict__ ids, float * __restrict__ weights, const int k, const float s_sum, const float b_sum,
        const float s_w, const float b_w, const int64_t ids_nb1) {
    __shared__ float bs[NE];
    __shared__ float wsel[16];
    const int t = threadIdx.x;
    logits += (int64_t) blockIdx.x*NE;          // one block per token
    ids    += (int64_t) blockIdx.x*ids_nb1;
    weights += (int64_t) blockIdx.x*k;
    const float p = 1.0f/(1.0f + expf(-logits[t]));
    const float b = p + bias[t];
    bs[t] = b;
    __syncthreads();
    int rank = 0;
#pragma unroll 8
    for (int j = 0; j < NE; ++j) {
        const float o = bs[j];
        rank += (o > b) || (o == b && j < t);
    }
    if (rank < k) {
        ids[rank] = t;
        wsel[rank] = p;
    }
    __syncthreads();
    if (t < k) {
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            sum += wsel[j];
        }
        sum = sum*s_sum + b_sum;
        weights[t] = (wsel[t]/sum)*s_w + b_w;
    }
}

static int musa_fuse_router(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = getenv("GGML_MUSA_FUSE_ROUTER") == nullptr || atoi(getenv("GGML_MUSA_FUSE_ROUTER")) != 0;
    if (!on || i + 11 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_op want[12] = { GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
                               GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE };
    ggml_tensor * const * nd = cgraph->nodes + i;
    for (int u = 0; u < 12; ++u) {
        if (nd[u]->op != want[u]) {
            return 0;
        }
    }
    const ggml_tensor * sig = nd[0], * prs = nd[1], * add = nd[2], * as = nd[3], * gr = nd[5], * wrs = nd[6], * sum = nd[7], * ssc = nd[8];
    const ggml_tensor * dv = nd[9], * nrs = nd[10];
    ggml_tensor * topk = nd[4], * out = nd[11];
    const ggml_tensor * logits = sig->src[0], * bias = add->src[1];
    const int64_t NE = logits->ne[0], k = topk->ne[0], Nt = logits->ne[1];
    if (ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || prs->src[0] != sig || add->src[0] != sig || as->src[0] != add ||
        ggml_get_op_params_i32(as, 0) != GGML_SORT_ORDER_DESC || topk->view_src != as || topk->view_offs != 0 ||
        gr->src[0] != prs || gr->src[1] != topk || wrs->src[0] != gr || sum->src[0] != wrs || ssc->src[0] != sum ||
        dv->src[0] != wrs || dv->src[1] != ssc || nrs->src[0] != dv || out->src[0] != nrs ||
        NE != 128 || Nt < 1 || Nt > 65535 || ggml_nelements(logits) != NE*Nt || logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits) ||
        bias->type != GGML_TYPE_F32 || ggml_nelements(bias) != NE || !ggml_is_contiguous(bias) ||
        as->type != GGML_TYPE_I32 || ggml_nelements(as) != NE*Nt || !ggml_is_contiguous(as) || k < 1 || k > 16 || ggml_nelements(topk) != k*Nt ||
        out->type != GGML_TYPE_F32 || ggml_nelements(out) != k*Nt || !ggml_is_contiguous(out)) {
        return 0;
    }
    // the top-k VIEW (ids for the expert matmuls) and the final weights leave the subgraph
    const int idx[12] = { i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7, i + 8, i + 9, i + 10, i + 11 };
    const int outs[2] = { i + 4, i + 11 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idx, 12, want, outs, 2)) {
        return 0;
    }
    // (single block: every thread reads its logit and bias before the first barrier, ids/weights are written after it,
    // so outputs sharing memory with the inputs are safe; they must not overlap each other)
    // (with several tokens the blocks run concurrently: then nothing may overlap)
    if (musa_overlap(as->data, ggml_nbytes(as), out->data, ggml_nbytes(out)) ||
        (Nt > 1 && !musa_disjoint_impl({ as, out }, { logits, bias }, __LINE__))) {
        return 0;
    }
    if (musa_skip_fused("router")) {
        return 11;
    }
    musa_router<128><<<(unsigned) Nt, 128, 0, ctx.stream()>>>((const float *) logits->data, (const float *) bias->data, (int32_t *) as->data,
        (float *) out->data, (int) k, ggml_get_op_params_f32(ssc, 0), ggml_get_op_params_f32(ssc, 1),
        ggml_get_op_params_f32(out, 0), ggml_get_op_params_f32(out, 1), as->nb[1]/sizeof(int32_t));
    return 11;
}

// row clip with a per-element scale in front: MUL(x, w) followed by the 8-node clip on the product  ->  1 kernel
static int musa_fuse_scaled_row_clip(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = getenv("GGML_MUSA_FUSE_CLIPW") == nullptr || atoi(getenv("GGML_MUSA_FUSE_CLIPW")) != 0;
    if (!on || i + 8 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor ** n = cgraph->nodes + i;
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL, GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL,
                                             GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL }, { i + 8 })) {
        return 0;
    }
    const ggml_tensor * mul = n[0], * x = mul->src[0], * w = mul->src[1];
    if (n[1]->src[0] != mul || n[2]->src[0] != n[1] || n[3]->src[0] != n[2] || n[4]->src[0] != n[3] || n[5]->src[0] != n[4] ||
        n[6]->src[0] != n[5] || n[6]->src[1] != n[4] || n[7]->src[0] != n[6] || n[8]->src[0] != mul || n[8]->src[1] != n[7]) {
        return 0;
    }
    ggml_tensor * dst = n[8];
    if (!musa_is_f32_rows(x) || !musa_is_f32_rows(dst) || !ggml_are_same_shape(x, dst) || !ggml_are_same_shape(x, w) ||
        w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || n[2]->ne[0] != 1) {
        return 0;
    }
    if (!musa_disjoint_impl({ dst }, { w }, __LINE__) || (dst->data != x->data && !musa_disjoint_impl({ dst }, { x }, __LINE__))) {
        return 0;
    }
    const float s  = ggml_get_op_params_f32(n[4], 0), b = ggml_get_op_params_f32(n[4], 1);
    const float c  = ggml_get_op_params_f32(n[5], 0);
    const float lo = ggml_get_op_params_f32(n[7], 0), hi = ggml_get_op_params_f32(n[7], 1);
    const int64_t nrows = x->ne[1]*x->ne[2]*x->ne[3];
    if (nrows > INT32_MAX || x->ne[0] > 128) {     // one element per thread
        return 0;
    }
    const int64_t f = sizeof(float);
    if (musa_skip_fused("clip")) {
        return 8;
    }
    musa_row_clip<128><<<(unsigned) nrows, 128, 0, ctx.stream()>>>((const float *) x->data, (float *) dst->data, (int) x->ne[0],
        (int) x->ne[1], (int) x->ne[2], x->nb[1]/f, x->nb[2]/f, x->nb[3]/f, dst->nb[1]/f, dst->nb[2]/f, dst->nb[3]/f,
        s, b, c, lo, hi, (const float *) w->data);
    return 8;
}

// PAD of the (constant) bias table right before the rel-bias chain: zero rows appended to dim 1 only
static int musa_fuse_pad_rel_bias(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = getenv("GGML_MUSA_FUSE_RELB_PAD") == nullptr || atoi(getenv("GGML_MUSA_FUSE_RELB_PAD")) != 0;
    if (!on) {
        return 0;
    }
    const ggml_tensor * pad = cgraph->nodes[i];
    if (i + 1 >= cgraph->n_nodes || cgraph->nodes[i + 1]->op != GGML_OP_MUL_MAT || cgraph->nodes[i + 1]->src[0] != pad) {
        return 0;
    }
    for (int k = 0; k < 9; ++k) {
        if (k != 3 && ggml_get_op_params_i32(pad, k) != 0) {
            return 0;
        }
    }
    int r = musa_fuse_rel_bias(ctx, cgraph, i + 1, pad);
    if (!r) {
        r = musa_fuse_rel_table(ctx, cgraph, i + 1, pad);
    }
    return r ? r + 1 : 0;
}

// ---- batch of independent row normalizations (Inkling: q scaled clip, k clip, v RMS norm) in one launch, with the
// following SET_ROWS of k / v into the f16 KV cache folded in. Units: [MUL(x, w)] + 8-node clip, or RMS_NORM (no weight).
#define MUSA_NB_MAXU 6
struct musa_norm_unit {
    const float * x; const float * w; const float * wc; float * dst; half * cache; const int64_t * idx;   // wc: per-column weight (rms)
    int64_t sx1, sd1, cache_nb1;          // row strides (elements)
    int ne0, nrows, kind;                 // kind 0 = clip, 1 = rms (optionally times wc)
    int rpt;                              // rows per token (cache stores: token = row / rpt)
    float s, b, c, lo, hi, eps;
};
struct musa_norm_batch { musa_norm_unit u[MUSA_NB_MAXU]; int b0[MUSA_NB_MAXU + 1]; int n; };

#define MUSA_NB_BS 256
#define MUSA_NB_EPT 12                    // elements per thread: rows up to 3072 wide
static __global__ void __launch_bounds__(MUSA_NB_BS) musa_norm_batch_k(const musa_norm_batch nb) {
    __shared__ float sq[MUSA_NB_BS];
    __shared__ float part[16];
    int ui = 0;
    while (ui + 1 < nb.n && (int) blockIdx.x >= nb.b0[ui + 1]) {
        ++ui;
    }
    const musa_norm_unit & U = nb.u[ui];
    const int r = blockIdx.x - nb.b0[ui], t = threadIdx.x;
    float v[MUSA_NB_EPT];
    float a = 0.0f;
#pragma unroll
    for (int k = 0; k < MUSA_NB_EPT; ++k) {
        const int e = t + k*MUSA_NB_BS;
        v[k] = 0.0f;
        if (e < U.ne0) {
            v[k] = U.x[(int64_t) r*U.sx1 + e];
            if (U.w) {
                v[k] *= U.w[(int64_t) r*U.ne0 + e];
            }
        }
        a = fmaf(v[k], v[k], a);
    }
    sq[t] = a;
    __syncthreads();
    if (t < 16) {
        float p = 0.0f;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            p += sq[t*16 + k];
        }
        part[t] = p;
    }
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        tot += part[k];
    }
    const float f = U.kind == 0 ? fminf(fmaxf(U.c / (U.s*sqrtf(tot) + U.b), U.lo), U.hi) : rsqrtf(tot/U.ne0 + U.eps);
#pragma unroll
    for (int k = 0; k < MUSA_NB_EPT; ++k) {
        const int e = t + k*MUSA_NB_BS;
        if (e < U.ne0) {
            const float y = U.wc ? (v[k]*f)*U.wc[e] : v[k]*f;
            if (U.dst) {
                U.dst[(int64_t) r*U.sd1 + e] = y;
            }
            if (U.cache) {
                U.cache[U.idx[r / U.rpt]*U.cache_nb1 + (int64_t) (r % U.rpt)*U.ne0 + e] = __float2half(y);
            }
        }
    }
}

static bool musa_is_viewish(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

// parse one unit starting at node j; returns the index of its last node (or -1); fills U and the computed node indices
static int musa_parse_norm_unit(ggml_cgraph * cgraph, int j, musa_norm_unit & U, std::vector<int> & idx, std::vector<ggml_op> & ops) {
    ggml_tensor * const * n = cgraph->nodes + j;
    const int nn = cgraph->n_nodes - j;
    if (n[0]->op == GGML_OP_RMS_NORM) {
        const ggml_tensor * x = n[0]->src[0];
        // rows up to 128 wide (one block per row), or one row of a multiple of 128 handled as 128-wide ... not supported:
        // whole-row norms of wider vectors use the 2304-wide variant below
        if (x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || x->ne[3] != 1 || x->nb[2] != x->nb[1]*x->ne[1] ||
            !ggml_is_contiguous(n[0])) {   // rows evenly strided over dims 1-2 (token batches: [ne0, heads, tokens])
            return -1;
        }
        // optional MUL by a per-column weight right after (build_norm): fused when the norm has no other reader
        const ggml_tensor * mw = nullptr;
        if (nn > 1 && n[1]->op == GGML_OP_MUL && n[1]->src[0] == n[0] && ggml_node_get_use_count(cgraph, j) == 1) {
            const ggml_tensor * w = n[1]->src[1];
            if (w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && ggml_nelements(w) == x->ne[0] && ggml_are_same_shape(n[1], n[0]) &&
                ggml_is_contiguous(n[1])) {
                mw = w;
            }
        }
        if (x->ne[0] > MUSA_NB_BS*MUSA_NB_EPT) {
            return -1;
        }
        U = {};
        U.x = (const float *) x->data; U.dst = (float *) (mw ? n[1] : n[0])->data; U.sx1 = x->nb[1]/sizeof(float); U.sd1 = n[0]->ne[0];
        U.ne0 = (int) x->ne[0]; U.nrows = (int) (x->ne[1]*x->ne[2]); U.rpt = (int) x->ne[1]; U.kind = 1; U.eps = ggml_get_op_params_f32(n[0], 0);
        U.wc = mw ? (const float *) mw->data : nullptr;
        idx.push_back(j); ops.push_back(GGML_OP_RMS_NORM);
        if (mw) {
            idx.push_back(j + 1); ops.push_back(GGML_OP_MUL);
            return j + 1;
        }
        return j;
    }
    const int off = n[0]->op == GGML_OP_MUL ? 1 : 0;
    if (nn < 8 + off) {
        return -1;
    }
    const ggml_op want[8] = { GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL };
    for (int u = 0; u < 8; ++u) {
        if (n[off + u]->op != want[u]) {
            return -1;
        }
    }
    ggml_tensor * const * c = n + off;
    const ggml_tensor * x = off ? n[0] : c[0]->src[0];          // the clipped tensor (the MUL result when scaled)
    if (c[0]->src[0] != x || c[1]->src[0] != c[0] || c[2]->src[0] != c[1] || c[3]->src[0] != c[2] || c[4]->src[0] != c[3] ||
        c[5]->src[0] != c[4] || c[5]->src[1] != c[3] || c[6]->src[0] != c[5] || c[7]->src[0] != x || c[7]->src[1] != c[6] ||
        c[1]->ne[0] != 1) {
        return -1;
    }
    const ggml_tensor * xin = off ? n[0]->src[0] : x, * w = off ? n[0]->src[1] : nullptr;
    const ggml_tensor * dst = c[7];
    if (xin->type != GGML_TYPE_F32 || xin->nb[0] != sizeof(float) || xin->ne[0] > MUSA_NB_BS*MUSA_NB_EPT || xin->ne[3] != 1 ||
        xin->nb[2] != xin->nb[1]*xin->ne[1] ||
        !ggml_is_contiguous(dst) || (w && (w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || !ggml_are_same_shape(w, xin)))) {
        return -1;
    }
    U = {};
    U.x = (const float *) xin->data; U.w = w ? (const float *) w->data : nullptr; U.dst = (float *) dst->data;
    U.sx1 = xin->nb[1]/sizeof(float); U.sd1 = dst->ne[0]; U.ne0 = (int) xin->ne[0]; U.nrows = (int) (xin->ne[1]*xin->ne[2]);
    U.rpt = (int) xin->ne[1]; U.kind = 0;
    U.s = ggml_get_op_params_f32(c[3], 0); U.b = ggml_get_op_params_f32(c[3], 1); U.c = ggml_get_op_params_f32(c[4], 0);
    U.lo = ggml_get_op_params_f32(c[6], 0); U.hi = ggml_get_op_params_f32(c[6], 1);
    for (int u = 0; u < 8 + off; ++u) {
        idx.push_back(j + u); ops.push_back(n[u]->op);
    }
    return j + 7 + off;
}

static int musa_fuse_norm_batch(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = getenv("GGML_MUSA_FUSE_NORMB") == nullptr || atoi(getenv("GGML_MUSA_FUSE_NORMB")) != 0;
    if (!on) {
        return 0;
    }
    musa_norm_batch B{};
    std::vector<int> idx, outs;
    std::vector<ggml_op> ops;
    std::vector<const ggml_tensor *> unit_dst, reads, wreads;
    int last = i - 1, j = i;
    while (B.n < MUSA_NB_MAXU && j < cgraph->n_nodes) {
        musa_norm_unit U;
        const size_t i0 = idx.size();
        const int e = musa_parse_norm_unit(cgraph, j, U, idx, ops);
        if (e < 0) {
            break;
        }
        // independence: a unit must not read another unit's result
        const ggml_tensor * dn = cgraph->nodes[e];
        bool dep = false;
        for (size_t k = i0; k < idx.size(); ++k) {
            const ggml_tensor * nd = cgraph->nodes[idx[k]];
            for (int q = 0; q < GGML_MAX_SRC && nd->src[q]; ++q) {
                for (const ggml_tensor * d : unit_dst) {
                    if (nd->src[q] == d || (nd->src[q]->view_src && nd->src[q]->view_src == d)) {
                        dep = true;
                    }
                }
            }
        }
        if (dep) {
            idx.resize(i0); ops.resize(i0);
            break;
        }
        reads.push_back(cgraph->nodes[idx[i0]]->src[0]);                    // x of this unit (index = unit number)
        wreads.push_back(U.w ? cgraph->nodes[idx[i0]]->src[1] : nullptr);
        {
            static const bool ud = getenv("GGML_MUSA_FUSE_DBG") != nullptr;
            static int np = 0;
            if (ud && np++ < 12) {
                fprintf(stderr, "norm_batch: anchor %d unit %d nodes %d..%d x=%s dst=%s kind=%d\n", i, B.n, idx[i0], e,
                        cgraph->nodes[idx[i0]]->src[0]->name, dn->name, U.kind);
            }
        }
        unit_dst.push_back(dn);
        outs.push_back(e);
        B.u[B.n++] = U;
        last = e;
        j = e + 1;
        while (j < cgraph->n_nodes && musa_is_viewish(cgraph->nodes[j])) {
            ++j;
        }
    }
    static const bool ndbg = getenv("GGML_MUSA_FUSE_DBG") != nullptr;
    if (B.n < 2) {
        static int np = 0;
        if (ndbg && B.n == 1 && np++ < 6) {
            fprintf(stderr, "norm_batch: only %d unit(s) at %d (%s), next node %s\n", B.n, i, cgraph->nodes[i]->name,
                    j < cgraph->n_nodes ? ggml_op_desc(cgraph->nodes[j]) : "-");
        }
        return 0;
    }
    // following SET_ROWS of a unit's result (through a 2D view of the whole result) into an f16 cache row
    while (j < cgraph->n_nodes) {
        ggml_tensor * sr = cgraph->nodes[j];
        if (sr->op != GGML_OP_SET_ROWS || sr->type != GGML_TYPE_F16) {
            break;
        }
        const ggml_tensor * src = sr->src[0], * ix = sr->src[1];
        int uu = -1;
        for (int k = 0; k < B.n; ++k) {
            if (src->view_src == unit_dst[k] || src == unit_dst[k]) {
                uu = k;
            }
        }
        const int64_t ntok_u = uu >= 0 ? B.u[uu].nrows / B.u[uu].rpt : 0;
        if (uu < 0 || B.u[uu].cache || ix->type != GGML_TYPE_I64 || ggml_nelements(ix) != ntok_u || !ggml_is_contiguous(ix) ||
            src->ne[1] != ntok_u || src->ne[0] != (int64_t) B.u[uu].ne0*B.u[uu].rpt || !ggml_is_contiguous(src) || src->view_offs != 0 ||
            sr->ne[0] != src->ne[0] || sr->nb[0] != sizeof(half)) {
            break;
        }
        B.u[uu].cache = (half *) sr->data;
        B.u[uu].idx = (const int64_t *) ix->data;
        B.u[uu].cache_nb1 = sr->nb[1]/sizeof(half);
        // the f32 result is not needed when its only reader is the view feeding this (folded) store
        int vi = -1;
        for (int q = j - 1; q > outs[uu]; --q) {
            if (cgraph->nodes[q] == src) {
                vi = q;
                break;
            }
        }
        if (src != unit_dst[uu] && vi >= 0 && ggml_node_get_use_count(cgraph, outs[uu]) == 1 && ggml_node_get_use_count(cgraph, vi) == 1) {
            B.u[uu].dst = nullptr;
        }
        last = j;
        ++j;
        while (j < cgraph->n_nodes && cgraph->nodes[j]->op == GGML_OP_VIEW && j + 1 < cgraph->n_nodes &&
               cgraph->nodes[j + 1]->op == GGML_OP_SET_ROWS) {
            ++j;
        }
    }
    // unit results are outputs (Qcur feeds attention); everything inside a unit must stay inside
    if (!ggml_can_fuse_subgraph_ext(cgraph, idx.data(), (int) idx.size(), ops.data(), outs.data(), (int) outs.size())) {
        static int np = 0;
        if (ndbg && np++ < 3) fprintf(stderr, "norm_batch: generic check failed (%d units)\n", B.n);
        return 0;
    }
    // a unit may write in place over its own x (same rows, each block reads its row before writing it); any other
    // overlap between a result and an input (or between results) would race across blocks
    for (int a = 0; a < B.n; ++a) {
        const ggml_tensor * d = unit_dst[a];
        if (!B.u[a].dst) {
            continue;
        }
        for (int b = 0; b < B.n; ++b) {
            const ggml_tensor * r = reads[b];
            const bool same_inplace = a == b && d->data == r->data && d->ne[0] == r->ne[0] && d->nb[1] == r->nb[1];
            if (!same_inplace && musa_overlap(d->data, ggml_nbytes(d), r->data, ggml_nbytes(r))) {
                static int np = 0;
                if (ndbg) fprintf(stderr, "norm_batch: unit %d result %s overlaps unit %d input %s%s\n", a, d->name, b, r->name,
                                              d->data == r->data ? " (same address)" : "");
                return 0;
            }
            if (wreads[b] && musa_overlap(d->data, ggml_nbytes(d), wreads[b]->data, ggml_nbytes(wreads[b]))) {
                if (ndbg) fprintf(stderr, "norm_batch: unit %d result %s overlaps unit %d weight\n", a, d->name, b);
                return 0;
            }
            if (b != a && B.u[b].dst && musa_overlap(d->data, ggml_nbytes(d), unit_dst[b]->data, ggml_nbytes(unit_dst[b]))) {
                if (ndbg) fprintf(stderr, "norm_batch: unit %d result %s overlaps unit %d result %s\n", a, d->name, b, unit_dst[b]->name);
                return 0;
            }
        }
    }
    if (musa_skip_fused("normb")) {
        return last - i;
    }
    int blk = 0;
    for (int k = 0; k < B.n; ++k) {
        B.b0[k] = blk;
        blk += B.u[k].nrows;
    }
    B.b0[B.n] = blk;
    {
        static int nf = 0;
        if (ndbg && (nf++ % 1000 == 0 || nf < 300)) fprintf(stderr, "norm_batch: %d fused (%d units, k cache %d, v cache %d)\n", nf, B.n,
                                              B.n > 1 && B.u[1].cache != nullptr, B.n > 2 && B.u[2].cache != nullptr);
    }
    musa_norm_batch_k<<<blk, MUSA_NB_BS, 0, ctx.stream()>>>(B);
    return last - i;
}

int ggml_cuda_musa_try_fuse(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = getenv("GGML_MUSA_FUSE") == nullptr || atoi(getenv("GGML_MUSA_FUSE")) != 0;
    if (!enabled) {
        return 0;
    }
    switch (cgraph->nodes[i]->op) {
        case GGML_OP_SQR: {
            if (const int r = musa_fuse_norm_batch(ctx, cgraph, i)) {
                return r;
            }
            return musa_fuse_row_clip(ctx, cgraph, i);
        }
        case GGML_OP_MUL: {
            if (const int r = musa_fuse_norm_batch(ctx, cgraph, i)) {
                return r;
            }
            return musa_fuse_scaled_row_clip(ctx, cgraph, i);
        }
        case GGML_OP_CONT: return musa_fuse_control_norm(ctx, cgraph, i);
        case GGML_OP_RMS_NORM: {
            if (const int r = musa_fuse_control_norm(ctx, cgraph, i)) {
                return r;
            }
            return musa_fuse_norm_batch(ctx, cgraph, i);
        }
        case GGML_OP_MUL_MAT: {
            if (const int r = musa_fuse_rel_bias(ctx, cgraph, i)) {
                return r;
            }
            if (const int r = musa_fuse_rel_table(ctx, cgraph, i)) {
                return r;
            }
            return ggml_cuda_musa_mm_group(ctx, cgraph, i);
        }
        case GGML_OP_SCALE:   return musa_fuse_situ(ctx, cgraph, i);
        case GGML_OP_UNARY:   return musa_fuse_router(ctx, cgraph, i);
        case GGML_OP_PAD:     return musa_fuse_pad_rel_bias(ctx, cgraph, i);
        case GGML_OP_CONCAT: {
            if (const int r = musa_fuse_control(ctx, cgraph, i)) {
                return r;
            }
            if (const int r = musa_fuse_short_conv(ctx, cgraph, i)) {
                return r;
            }
            return musa_fuse_mix(ctx, cgraph, i);
        }
        default:          return 0;
    }
}

// ---- small copies / concat without 64-bit index divisions (upstream's per-element kernels divide the flat index
// into 4 coordinates; integer division is very slow on mp_21 and these ops are tiny: 25-45 us -> a few us)
template <typename TS, typename TD>
static __global__ void musa_cpy_same_shape(const char * __restrict__ src, char * __restrict__ dst, const int ne0, const int ne2,
        const int64_t sn0, const int64_t sn1, const int64_t sn2, const int64_t sn3,
        const int64_t dn0, const int64_t dn1, const int64_t dn2, const int64_t dn3) {
    const int i1 = blockIdx.y, i2 = blockIdx.z % ne2, i3 = blockIdx.z / ne2;
    const char * s = src + i1*sn1 + i2*sn2 + i3*sn3;
    char * d = dst + i1*dn1 + i2*dn2 + i3*dn3;
    for (int i0 = blockIdx.x*blockDim.x + threadIdx.x; i0 < ne0; i0 += gridDim.x*blockDim.x) {
        const TS v = *(const TS *) (s + i0*sn0);
        *(TD *) (d + i0*dn0) = ggml_cuda_cast<TD>(v);
    }
}

// transposed f32 copies (src contiguous along dim 1, dst contiguous): 32x32 tiles through shared memory so that both
// the reads and the writes are coalesced. The generic same-shape kernel launches one 128-thread block per row, which for
// Inkling's [k = 2..5, 2304, n_tokens] transposes means ~300k blocks with 2-5 active threads.
static __global__ void __launch_bounds__(256) musa_transpose_f32(const char * __restrict__ src, float * __restrict__ dst,
        const int ne0, const int ne1, const int64_t snb0, const int64_t snb2) {
    __shared__ float tile[32][33];
    const int i1b = blockIdx.x*32, i0b = blockIdx.y*32, i2 = blockIdx.z;
    const char * s2 = src + (int64_t) i2*snb2;
    float * d2 = dst + (int64_t) i2*ne0*ne1;
    // read: threadIdx.x along dim 1 (contiguous in src)
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int i0 = i0b + r, i1 = i1b + threadIdx.x;
        if (i0 < ne0 && i1 < ne1) {
            tile[r][threadIdx.x] = ((const float *) (s2 + (int64_t) i0*snb0))[i1];
        }
    }
    __syncthreads();
    // write: threadIdx.x along dim 0 (contiguous in dst)
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int i1 = i1b + r, i0 = i0b + threadIdx.x;
        if (i0 < ne0 && i1 < ne1) {
            d2[(int64_t) i1*ne0 + i0] = tile[threadIdx.x][r];
        }
    }
}

bool ggml_cuda_musa_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_MUSA_CPY_FAST") == nullptr || atoi(getenv("GGML_MUSA_CPY_FAST")) != 0;
    const bool ts_ok = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16;
    const bool td_ok = dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16;
    if (!enabled || !ts_ok || !td_ok || !ggml_are_same_shape(src, dst) || src->ne[0] > INT32_MAX ||
        src->ne[1] > 65535 || src->ne[2]*src->ne[3] > 65535) {
        return false;
    }
    const int ne0 = (int) src->ne[0];
    cudaStream_t st = ctx.stream();
    static const bool tr_on = getenv("GGML_MUSA_CPY_TRANSPOSE") == nullptr || atoi(getenv("GGML_MUSA_CPY_TRANSPOSE")) != 0;
    if (tr_on && src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && src->nb[1] == sizeof(float) &&
        src->nb[0] != sizeof(float) && src->ne[3] == 1 && src->ne[2] <= 65535 && src->ne[1] <= INT32_MAX &&
        ggml_nelements(src) >= 65536) {          // small (decode) transposes stay with the generic kernel (faster there)
        const dim3 tg((unsigned) ((src->ne[1] + 31)/32), (unsigned) ((ne0 + 31)/32), (unsigned) src->ne[2]);
        musa_transpose_f32<<<tg, dim3(32, 8), 0, st>>>((const char *) src->data, (float *) dst->data, ne0, (int) src->ne[1], src->nb[0], src->nb[2]);
        return true;
    }
    const dim3 grid((unsigned) std::min<int64_t>((ne0 + 127)/128, 64), (unsigned) src->ne[1], (unsigned) (src->ne[2]*src->ne[3]));
#define MUSA_CPY_LAUNCH(TS, TD) musa_cpy_same_shape<TS, TD><<<grid, 128, 0, st>>>((const char *) src->data, (char *) dst->data, ne0, \
        (int) src->ne[2], src->nb[0], src->nb[1], src->nb[2], src->nb[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3])
    if (src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32)      { MUSA_CPY_LAUNCH(float, float); }
    else if (src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) { MUSA_CPY_LAUNCH(float, half);  }
    else if (src->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) { MUSA_CPY_LAUNCH(half, float);  }
    else                                                               { MUSA_CPY_LAUNCH(half, half);   }
#undef MUSA_CPY_LAUNCH
    return true;
}

static __global__ void musa_concat_f32(const char * __restrict__ a, const char * __restrict__ b, char * __restrict__ dst,
        const int dim, const int na, const int ne0, const int ne2,
        const int64_t an0, const int64_t an1, const int64_t an2, const int64_t an3,
        const int64_t bn0, const int64_t bn1, const int64_t bn2, const int64_t bn3,
        const int64_t dn0, const int64_t dn1, const int64_t dn2, const int64_t dn3) {
    const int i1 = blockIdx.y, i2 = blockIdx.z % ne2, i3 = blockIdx.z / ne2;
    for (int i0 = blockIdx.x*blockDim.x + threadIdx.x; i0 < ne0; i0 += gridDim.x*blockDim.x) {
        int c[4] = {i0, i1, i2, i3};
        const char * s;
        if (c[dim] < na) {
            s = a + c[0]*an0 + c[1]*an1 + c[2]*an2 + c[3]*an3;
        } else {
            c[dim] -= na;
            s = b + c[0]*bn0 + c[1]*bn1 + c[2]*bn2 + c[3]*bn3;
        }
        *(float *) (dst + i0*dn0 + i1*dn1 + i2*dn2 + i3*dn3) = *(const float *) s;
    }
}

bool ggml_cuda_musa_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_MUSA_CPY_FAST") == nullptr || atoi(getenv("GGML_MUSA_CPY_FAST")) != 0;
    const ggml_tensor * a = dst->src[0], * b = dst->src[1];
    const int dim = ggml_get_op_params_i32(dst, 0);
    if (!enabled || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || dim < 0 || dim > 3 ||
        dst->ne[0] > INT32_MAX || dst->ne[1] > 65535 || dst->ne[2]*dst->ne[3] > 65535) {
        return false;
    }
    const int ne0 = (int) dst->ne[0];
    const dim3 grid((unsigned) std::min<int64_t>((ne0 + 127)/128, 64), (unsigned) dst->ne[1], (unsigned) (dst->ne[2]*dst->ne[3]));
    musa_concat_f32<<<grid, 128, 0, ctx.stream()>>>((const char *) a->data, (const char *) b->data, (char *) dst->data,
        dim, (int) a->ne[dim], ne0, (int) dst->ne[2],
        a->nb[0], a->nb[1], a->nb[2], a->nb[3], b->nb[0], b->nb[1], b->nb[2], b->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    return true;
}

// ---- tiny Q8_0 matmuls (Inkling: sink projection 24 rows, controller down 64 rows x 4608, controller gain/forget
// 2304 rows x 64): f32 activations, weights in their original layout, N <= 4 columns. Thread group of TPR threads per
// row (TPR = 128 for long rows), 128/TPR rows per block; partial sums reduced in shared memory.
template <int TPR, int NC>
static __global__ void musa_q8_small(const block_q8_0 * __restrict__ w, const float * __restrict__ x, float * __restrict__ d,
        const int R, const int nb, const int64_t w_row_blocks, const int64_t x_col, const int64_t d_col) {
    constexpr int RPB = 128/TPR;
    __shared__ float red[NC][128];
    x += (int64_t) blockIdx.y*NC*x_col;      // column chunk (prompt batches: grid.y = N/NC)
    d += (int64_t) blockIdx.y*NC*d_col;
    const int t = threadIdx.x % TPR, rr = threadIdx.x / TPR;
    const int r = blockIdx.x*RPB + rr;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        acc[c] = 0.0f;
    }
    if (r < R) {
        const block_q8_0 * wr = w + (int64_t) r*w_row_blocks;
        for (int b = t; b < nb; b += TPR) {
            const float dw = __half2float(wr[b].d);
            const int8_t * q = wr[b].qs;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const float * xb = x + c*x_col + b*32;
                float s = 0.0f;
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    s = fmaf((float) q[i], xb[i], s);
                }
                acc[c] = fmaf(dw, s, acc[c]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        red[c][threadIdx.x] = acc[c];
    }
    __syncthreads();
#pragma unroll
    for (int w2 = TPR/2; w2 > 0; w2 >>= 1) {
        if (t < w2) {
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                red[c][threadIdx.x] += red[c][threadIdx.x + w2];
            }
        }
        __syncthreads();
    }
    if (t == 0 && r < R) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            d[c*d_col + r] = red[c][threadIdx.x];
        }
    }
}

bool ggml_cuda_musa_q8_small_eligible(const ggml_tensor * src0) {
    static const bool enabled = getenv("GGML_MUSA_Q8_SMALL") == nullptr || atoi(getenv("GGML_MUSA_Q8_SMALL")) != 0;
    return enabled && src0->type == GGML_TYPE_Q8_0 && src0->ne[2] == 1 && src0->ne[3] == 1 && src0->ne[0] % 32 == 0 &&
           src0->nb[0] == sizeof(block_q8_0) && (src0->ne[1] <= 128 || src0->ne[0] <= 256);
}

bool ggml_cuda_musa_q8_small(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (!ggml_cuda_musa_q8_small_eligible(src0)) {
        return false;
    }
    const int64_t K = src0->ne[0], R = src0->ne[1], N = src1->ne[1];
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->ne[0] != K || N > 512 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
        src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || dst->ne[0] != R || dst->ne[1] != N) {
        return false;
    }
    const int nb = (int) (K/32);
    const int64_t wrb = src0->nb[1]/sizeof(block_q8_0), xc = src1->nb[1]/sizeof(float), dc = dst->nb[1]/sizeof(float);
    cudaStream_t st = ctx.stream();
    const block_q8_0 * w = (const block_q8_0 *) src0->data;
    const float * x = (const float *) src1->data;
    float * d = (float *) dst->data;
    // full 4-column chunks over grid.y, then the 1-3 remaining columns
    const int nfull = (int) (N/4), rem = (int) (N%4);
    const float * xr = x + (int64_t) nfull*4*xc;
    float * dr = d + (int64_t) nfull*4*dc;
#define Q8S_LAUNCH(TPR) do { const int rpb = 128/(TPR); const unsigned gx = (unsigned) ((R + rpb - 1)/rpb); \
        if (nfull > 0) { musa_q8_small<TPR, 4><<<dim3(gx, (unsigned) nfull), 128, 0, st>>>(w, x, d, (int) R, nb, wrb, xc, dc); } \
        switch (rem) { \
            case 1: musa_q8_small<TPR, 1><<<gx, 128, 0, st>>>(w, xr, dr, (int) R, nb, wrb, xc, dc); break; \
            case 2: musa_q8_small<TPR, 2><<<gx, 128, 0, st>>>(w, xr, dr, (int) R, nb, wrb, xc, dc); break; \
            case 3: musa_q8_small<TPR, 3><<<gx, 128, 0, st>>>(w, xr, dr, (int) R, nb, wrb, xc, dc); break; \
            default: break; \
        } } while (0)
    if (nb >= 64)      { Q8S_LAUNCH(128); }
    else if (nb >= 16) { Q8S_LAUNCH(16); }
    else if (nb >= 4)  { Q8S_LAUNCH(4); }
    else               { Q8S_LAUNCH(1); }
#undef Q8S_LAUNCH
    return true;
}

// ---- single-token attention for small heads groups with per-head masks and sinks (Inkling decode):
// Q f32 [128, 1, Hq], K/V f16 [128, n_kv, Hkv] (cache views), mask f16 [n_kv, 1, Hq or 1], sinks f32 [Hq] (optional).
// kernel 1: block (kv head, 128-position chunk): lanes = positions for the scores, lanes = dims for P*V; G query heads
// of the group share each K/V load. kernel 2: combine chunks + sink per query head.
#define MUSA_FAD_D 128
#define MUSA_FAD_CH 32
#define MUSA_FAD_KP (MUSA_FAD_D + 8)        // padded K tile row (halves): conflict-free column reads
template <int G>
static __global__ void __launch_bounds__(128) musa_fa_dec_part(const char * __restrict__ Q, const int64_t q_nb2,
        const char * __restrict__ K, const int64_t k_nb1, const int64_t k_nb2, const char * __restrict__ V, const int64_t v_nb1, const int64_t v_nb2,
        const char * __restrict__ mask, const int64_t m_nb2, const int n_kv, const float scale,
        float * __restrict__ part_o, float2 * __restrict__ part_ml, const int nsplit) {
    __shared__ float qs[G][MUSA_FAD_D];
    __shared__ __align__(16) half kt[MUSA_FAD_CH][MUSA_FAD_KP];
    __shared__ __align__(16) half vt[MUSA_FAD_CH][MUSA_FAD_KP];
    __shared__ float half_sc[G][4][MUSA_FAD_CH];
    __shared__ float ps[G][MUSA_FAD_CH];
    __shared__ float red[G][2];
    const int hk = blockIdx.x, sp = blockIdx.y, l = threadIdx.x;
    const int p0 = sp*MUSA_FAD_CH, cnt = min(MUSA_FAD_CH, n_kv - p0);
#pragma unroll
    for (int g = 0; g < G; ++g) {
        qs[g][l] = ((const float *) (Q + (int64_t) (hk*G + g)*q_nb2))[l]*scale;
    }
    // K chunk -> shared memory, coalesced: each row is 64 x 4 bytes; 128 threads load 2 rows per step
    {
        // K and V chunks -> shared memory with 16-byte loads (load instructions are expensive on mp_21)
        const int c = l & 15, r0 = l >> 4;                 // 16 uint4 per row, 8 rows per step
        uint4 kv[MUSA_FAD_CH/8], vv4[MUSA_FAD_CH/8];
#pragma unroll
        for (int u = 0; u < MUSA_FAD_CH/8; ++u) {
            const int r = r0 + 8*u;
            const bool ok = r < cnt;
            kv[u]  = ok ? ((const uint4 *) (K + (int64_t) (p0 + r)*k_nb1 + (int64_t) hk*k_nb2))[c] : make_uint4(0, 0, 0, 0);
            vv4[u] = ok ? ((const uint4 *) (V + (int64_t) (p0 + r)*v_nb1 + (int64_t) hk*v_nb2))[c] : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < MUSA_FAD_CH/8; ++u) {
            *(uint4 *) &kt[r0 + 8*u][8*c] = kv[u];
            *(uint4 *) &vt[r0 + 8*u][8*c] = vv4[u];
        }
    }
    __syncthreads();
    // scores: thread = (position l&63, half of the dims l>>6)
    {
        const int pp = l & 31, hh = l >> 5;
        float a[G];
#pragma unroll
        for (int g = 0; g < G; ++g) a[g] = 0.0f;
        const half * kr = &kt[pp][hh*32];
#pragma unroll
        for (int d = 0; d < 32; d += 2) {
            const float2 kv = __half22float2(*(const half2 *) (kr + d));
#pragma unroll
            for (int g = 0; g < G; ++g) {
                a[g] = fmaf(qs[g][hh*32 + d], kv.x, fmaf(qs[g][hh*32 + d + 1], kv.y, a[g]));
            }
        }
#pragma unroll
        for (int g = 0; g < G; ++g) half_sc[g][hh][pp] = a[g];
    }
    __syncthreads();
    // lanes < 64: full score, mask; per-head chunk max / sum through shared memory
    float sc[G];
    if (l < MUSA_FAD_CH) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            sc[g] = l < cnt ? ((half_sc[g][0][l] + half_sc[g][1][l]) + (half_sc[g][2][l] + half_sc[g][3][l])) +
                              __half2float(((const half *) (mask + (int64_t) (hk*G + g)*m_nb2))[p0 + l]) : -INFINITY;
            ps[g][l] = sc[g];
        }
    }
    __syncthreads();
    if (l < G) {
        float m = -INFINITY;
        for (int u = 0; u < MUSA_FAD_CH; ++u) m = fmaxf(m, ps[l][u]);
        red[l][0] = m;
    }
    __syncthreads();
    if (l < MUSA_FAD_CH) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float m = red[g][0];
            ps[g][l] = m == -INFINITY ? 0.0f : expf(sc[g] - m);
        }
    }
    __syncthreads();
    if (l < G) {
        float t = 0.0f;
        for (int u = 0; u < MUSA_FAD_CH; ++u) t += ps[l][u];
        red[l][1] = t;
    }
    // P*V: lane = dim, coalesced V rows
    float o[G];
#pragma unroll
    for (int g = 0; g < G; ++g) o[g] = 0.0f;
#pragma unroll
    for (int j = 0; j < MUSA_FAD_CH; ++j) {
        const float v = __half2float(vt[j][l]);
#pragma unroll
        for (int g = 0; g < G; ++g) o[g] = fmaf(ps[g][j], v, o[g]);
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const int64_t slot = (int64_t) (hk*G + g)*nsplit + sp;
        part_o[slot*MUSA_FAD_D + l] = o[g];
        if (l == 0) {
            part_ml[slot] = make_float2(red[g][0], red[g][1]);
        }
    }
}

static __global__ void __launch_bounds__(128) musa_fa_dec_combine(const float * __restrict__ part_o, const float2 * __restrict__ part_ml,
        const float * __restrict__ sinks, float * __restrict__ dst, const int nsplit) {
    const int h = blockIdx.x, l = threadIdx.x;
    float M = sinks ? sinks[h] : -INFINITY;
    for (int s = 0; s < nsplit; ++s) {
        M = fmaxf(M, part_ml[(int64_t) h*nsplit + s].x);
    }
    float den = sinks ? expf(sinks[h] - M) : 0.0f, acc = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float2 ml = part_ml[(int64_t) h*nsplit + s];
        const float w = ml.x == -INFINITY ? 0.0f : expf(ml.x - M);
        den = fmaf(w, ml.y, den);
        acc = fmaf(w, part_o[((int64_t) h*nsplit + s)*MUSA_FAD_D + l], acc);
    }
    dst[(int64_t) h*MUSA_FAD_D + l] = acc/den;
}


bool ggml_cuda_musa_fa_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->src[6]) { return false; } // in-attention relative-position bias: upstream kernels only
    // default off: on par with the ggml vector kernel at Inkling's shapes (latency-bound, 1 block/MP); GGML_MUSA_FA_DEC=1 enables
    static const bool on = getenv("GGML_MUSA_FA_DEC") != nullptr && atoi(getenv("GGML_MUSA_FA_DEC")) != 0;
    if (!on) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (!Q || !K || !V || !mask || dst->src[5] || max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    const int64_t D = Q->ne[0], Hq = Q->ne[2], Hkv = K->ne[2], n_kv = K->ne[1];
    if (D != MUSA_FAD_D || K->ne[0] != D || V->ne[0] != D || Q->ne[1] != 1 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 ||
        Q->type != GGML_TYPE_F32 || Q->nb[0] != sizeof(float) || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 ||
        K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) || V->ne[1] != n_kv || V->ne[2] != Hkv || Hq % Hkv != 0 ||
        mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(half) || mask->ne[0] < n_kv || mask->ne[3] != 1 ||
        (mask->ne[2] != 1 && mask->ne[2] != Hq) ||
        (sinks && (sinks->type != GGML_TYPE_F32 || ggml_nelements(sinks) != Hq || !ggml_is_contiguous(sinks))) ||
        dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || ggml_nelements(dst) != D*Hq || (K->nb[1] % 4) != 0 ||
        ((uintptr_t) K->data % 4) != 0 || n_kv > INT32_MAX) {
        return false;
    }
    const int G = (int) (Hq/Hkv);
    if (G != 1 && G != 2 && G != 3 && G != 4) {
        return false;
    }
    const int nsplit = (int) ((n_kv + MUSA_FAD_CH - 1)/MUSA_FAD_CH);
    ggml_cuda_pool_alloc<float>  po(ctx.pool(), (size_t) Hq*nsplit*MUSA_FAD_D);
    ggml_cuda_pool_alloc<float2> pml(ctx.pool(), (size_t) Hq*nsplit);
    cudaStream_t st = ctx.stream();
    const int64_t m_nb2 = mask->ne[2] == 1 ? 0 : (int64_t) mask->nb[2];
    const dim3 grid((unsigned) Hkv, (unsigned) nsplit);
#define FAD_LAUNCH(GG) musa_fa_dec_part<GG><<<grid, 128, 0, st>>>((const char *) Q->data, Q->nb[2], (const char *) K->data, K->nb[1], K->nb[2], \
        (const char *) V->data, V->nb[1], V->nb[2], (const char *) mask->data, m_nb2, (int) n_kv, scale, po.get(), pml.get(), nsplit)
    switch (G) {
        case 1: FAD_LAUNCH(1); break;
        case 2: FAD_LAUNCH(2); break;
        case 3: FAD_LAUNCH(3); break;
        default: FAD_LAUNCH(4); break;
    }
#undef FAD_LAUNCH
    musa_fa_dec_combine<<<(unsigned) Hq, 128, 0, st>>>(po.get(), pml.get(), sinks ? (const float *) sinks->data : nullptr,
        (float *) dst->data, nsplit);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ---- attention for 2-4 query tokens (MTP verify), D = 128, f16 K/V, per-head masks, sinks: the R = G*NQ query rows of
// a KV head share each 32-position K/V tile (staged with 16-byte loads); split over position chunks + combine.
#define MUSA_FAS_CH 32
template <int G, int NQ>
static __global__ void __launch_bounds__(128) musa_fa_small_part(const char * __restrict__ Q, const int64_t q_nb1, const int64_t q_nb2,
        const char * __restrict__ K, const int64_t k_nb1, const int64_t k_nb2, const char * __restrict__ V, const int64_t v_nb1, const int64_t v_nb2,
        const char * __restrict__ mask, const int64_t m_nb1, const int64_t m_nb2, const int n_kv, const int Hq, const float scale,
        float * __restrict__ part_o, float2 * __restrict__ part_ml, const int nsplit) {
    constexpr int R = G*NQ, D = 128, CH = MUSA_FAS_CH;
    __shared__ float qs[R][D];
    __shared__ __align__(16) half kt[CH][D + 8];
    __shared__ __align__(16) half vt[CH][D];
    __shared__ float sc[R][CH];
    __shared__ float ml[R][2];
    const int hk = blockIdx.x, sp = blockIdx.y, l = threadIdx.x;
    const int p0 = sp*CH, cnt = min(CH, n_kv - p0);
    // query rows r = t*G + g (token t, head hk*G + g)
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int t = r / G, g = r % G;
        qs[r][l] = ((const float *) (Q + (int64_t) t*q_nb1 + (int64_t) (hk*G + g)*q_nb2))[l]*scale;
    }
    {
        const int c = l & 15, r0 = l >> 4;
        uint4 kv[CH/8], vv[CH/8];
#pragma unroll
        for (int u = 0; u < CH/8; ++u) {
            const int r = r0 + 8*u;
            const bool ok = r < cnt;
            kv[u] = ok ? ((const uint4 *) (K + (int64_t) (p0 + r)*k_nb1 + (int64_t) hk*k_nb2))[c] : make_uint4(0, 0, 0, 0);
            vv[u] = ok ? ((const uint4 *) (V + (int64_t) (p0 + r)*v_nb1 + (int64_t) hk*v_nb2))[c] : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < CH/8; ++u) {
            *(uint4 *) &kt[r0 + 8*u][8*c] = kv[u];
            *(uint4 *) &vt[r0 + 8*u][8*c] = vv[u];
        }
    }
    __syncthreads();
    {   // scores: position pp = l & 31, rows rg, rg + 4, ...
        const int pp = l & 31, rg = l >> 5;
        constexpr int RPT = (R + 3)/4;
        float a[RPT];
#pragma unroll
        for (int i = 0; i < RPT; ++i) a[i] = 0.0f;
        const half * kr = &kt[pp][0];
#pragma unroll 8
        for (int d = 0; d < D; d += 2) {
            const float2 kf = __half22float2(*(const half2 *) (kr + d));
#pragma unroll
            for (int i = 0; i < RPT; ++i) {
                const int r = rg + 4*i;
                if (r < R) {
                    a[i] = fmaf(qs[r][d], kf.x, fmaf(qs[r][d + 1], kf.y, a[i]));
                }
            }
        }
#pragma unroll
        for (int i = 0; i < RPT; ++i) {
            const int r = rg + 4*i;
            if (r < R) {
                const int t = r / G, h = hk*G + r % G;
                sc[r][pp] = pp < cnt ? a[i] + __half2float(((const half *) (mask + (int64_t) t*m_nb1 + (int64_t) h*m_nb2))[p0 + pp]) : -INFINITY;
            }
        }
    }
    __syncthreads();
    if (l < R) {
        float m = -INFINITY;
#pragma unroll 8
        for (int u = 0; u < CH; ++u) m = fmaxf(m, sc[l][u]);
        float s = 0.0f;
#pragma unroll 8
        for (int u = 0; u < CH; ++u) {
            const float e = m == -INFINITY ? 0.0f : expf(sc[l][u] - m);
            sc[l][u] = e;
            s += e;
        }
        ml[l][0] = m; ml[l][1] = s;
    }
    __syncthreads();
    float o[R];
#pragma unroll
    for (int r = 0; r < R; ++r) o[r] = 0.0f;
#pragma unroll 4
    for (int j = 0; j < CH; ++j) {
        const float v = __half2float(vt[j][l]);
#pragma unroll
        for (int r = 0; r < R; ++r) o[r] = fmaf(sc[r][j], v, o[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int t = r / G, h = hk*G + r % G;
        const int64_t slot = ((int64_t) t*Hq + h)*nsplit + sp;
        part_o[slot*D + l] = o[r];
        if (l == 0) {
            part_ml[slot] = make_float2(ml[r][0], ml[r][1]);
        }
    }
}

// rows = token*Hq + head; dst [D, Hq, NQ] contiguous
static __global__ void __launch_bounds__(128) musa_fa_small_combine(const float * __restrict__ part_o, const float2 * __restrict__ part_ml,
        const float * __restrict__ sinks, float * __restrict__ dst, const int nsplit, const int Hq) {
    const int row = blockIdx.x, h = row % Hq, l = threadIdx.x;
    float M = sinks ? sinks[h] : -INFINITY;
    for (int s = 0; s < nsplit; ++s) {
        M = fmaxf(M, part_ml[(int64_t) row*nsplit + s].x);
    }
    float den = sinks ? expf(sinks[h] - M) : 0.0f, acc = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float2 m2 = part_ml[(int64_t) row*nsplit + s];
        const float w = m2.x == -INFINITY ? 0.0f : expf(m2.x - M);
        den = fmaf(w, m2.y, den);
        acc = fmaf(w, part_o[((int64_t) row*nsplit + s)*128 + l], acc);
    }
    dst[(int64_t) row*128 + l] = acc/den;
}

bool ggml_cuda_musa_fa_small(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->src[6]) { return false; } // in-attention relative-position bias: upstream kernels only
    static const bool on = getenv("GGML_MUSA_FA_SMALL") != nullptr && atoi(getenv("GGML_MUSA_FA_SMALL")) != 0; // off: measured slower than upstream tile (d1024 N4: 146.7 vs 135.5 ms)
    if (!on) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (!Q || !K || !V || !mask || dst->src[5] || max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    const int64_t D = Q->ne[0], NQ = Q->ne[1], Hq = Q->ne[2], Hkv = K->ne[2], n_kv = K->ne[1];
    if (D != 128 || NQ < 2 || NQ > 4 || K->ne[0] != D || V->ne[0] != D || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 ||
        Q->type != GGML_TYPE_F32 || Q->nb[0] != sizeof(float) || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 ||
        K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) || V->ne[1] != n_kv || V->ne[2] != Hkv || Hq % Hkv != 0 ||
        mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(half) || mask->ne[0] < n_kv || mask->ne[1] < NQ || mask->ne[3] != 1 ||
        (mask->ne[2] != 1 && mask->ne[2] != Hq) ||
        (sinks && (sinks->type != GGML_TYPE_F32 || ggml_nelements(sinks) != Hq || !ggml_is_contiguous(sinks))) ||
        dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || ggml_nelements(dst) != D*Hq*NQ ||
        (K->nb[1] % 16) != 0 || (V->nb[1] % 16) != 0 || ((uintptr_t) K->data % 16) != 0 || ((uintptr_t) V->data % 16) != 0 ||
        (K->nb[2] % 16) != 0 || (V->nb[2] % 16) != 0 || n_kv > INT32_MAX) {
        return false;
    }
    const int G = (int) (Hq/Hkv);
    if (G != 3 && G != 2 && G != 1) {
        return false;
    }
    const int nsplit = (int) ((n_kv + MUSA_FAS_CH - 1)/MUSA_FAS_CH);
    ggml_cuda_pool_alloc<float>  po(ctx.pool(), (size_t) Hq*NQ*nsplit*D);
    ggml_cuda_pool_alloc<float2> pml(ctx.pool(), (size_t) Hq*NQ*nsplit);
    cudaStream_t st = ctx.stream();
    const int64_t m_nb2 = mask->ne[2] == 1 ? 0 : (int64_t) mask->nb[2];
    const dim3 grid((unsigned) Hkv, (unsigned) nsplit);
#define FAS_LAUNCH(GG, NN) musa_fa_small_part<GG, NN><<<grid, 128, 0, st>>>((const char *) Q->data, Q->nb[1], Q->nb[2], \
        (const char *) K->data, K->nb[1], K->nb[2], (const char *) V->data, V->nb[1], V->nb[2], (const char *) mask->data, mask->nb[1], m_nb2, \
        (int) n_kv, (int) Hq, scale, po.get(), pml.get(), nsplit)
#define FAS_G(GG) switch (NQ) { case 2: FAS_LAUNCH(GG, 2); break; case 3: FAS_LAUNCH(GG, 3); break; default: FAS_LAUNCH(GG, 4); break; }
    switch (G) {
        case 3: FAS_G(3); break;
        case 2: FAS_G(2); break;
        default: FAS_G(1); break;
    }
#undef FAS_G
#undef FAS_LAUNCH
    musa_fa_small_combine<<<(unsigned) (Hq*NQ), 128, 0, st>>>(po.get(), pml.get(), sinks ? (const float *) sinks->data : nullptr,
        (float *) dst->data, nsplit, (int) Hq);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
