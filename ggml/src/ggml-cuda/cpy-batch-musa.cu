// MTT S80: batch consecutive small f32 copies into one launch.
// With MTP speculative decoding every DeltaNet layer copies its conv state into K rollback slots as K separate
// GGML_OP_CPY nodes (~190 tiny launches per step on the 27B model). Kernel launches are the cost there, not bytes.
#include "cpy-batch-musa.cuh"

#define CPYB_MAX   8
#define CPYB_BLOCK 256
#define CPYB_MAX_ELEMS (1 << 20)

namespace {

struct cpyb_desc {
    const char * src;
    char       * dst;
    int64_t ne[4];
    int64_t nbs[4];
    int64_t nbd[4];
    int64_t n;
};
struct cpyb_args {
    cpyb_desc d[CPYB_MAX];
};

__global__ void __launch_bounds__(CPYB_BLOCK) cpyb_kernel(const cpyb_args a) {
    const cpyb_desc & d = a.d[blockIdx.y];
    for (int64_t i = (int64_t) blockIdx.x*CPYB_BLOCK + threadIdx.x; i < d.n; i += (int64_t) gridDim.x*CPYB_BLOCK) {
        int64_t r = i;
        const int64_t i0 = r % d.ne[0]; r /= d.ne[0];
        const int64_t i1 = r % d.ne[1]; r /= d.ne[1];
        const int64_t i2 = r % d.ne[2];
        const int64_t i3 = r / d.ne[2];
        *(float *) (d.dst + i0*d.nbd[0] + i1*d.nbd[1] + i2*d.nbd[2] + i3*d.nbd[3]) =
            *(const float *) (d.src + i0*d.nbs[0] + i1*d.nbs[1] + i2*d.nbs[2] + i3*d.nbs[3]);
    }
}

bool cpyb_is_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE;
}

bool cpyb_eligible(const ggml_tensor * t) {
    return t->op == GGML_OP_CPY && t->src[0] != nullptr && t->src[1] != nullptr &&
           t->src[0]->type == GGML_TYPE_F32 && t->src[1]->type == GGML_TYPE_F32 && t->type == GGML_TYPE_F32 &&
           ggml_nelements(t->src[0]) == ggml_nelements(t->src[1]) && ggml_nelements(t->src[0]) <= CPYB_MAX_ELEMS &&
           (t->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
}

// [lo, hi) byte span touched by a (possibly strided) tensor
void cpyb_span(const ggml_tensor * t, uintptr_t & lo, uintptr_t & hi) {
    lo = (uintptr_t) t->data;
    hi = lo + ggml_nbytes(t);
}

bool cpyb_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    uintptr_t al, ah, bl, bh;
    cpyb_span(a, al, ah);
    cpyb_span(b, bl, bh);
    return al < bh && bl < ah;
}

} // namespace

int ggml_cuda_musa_try_batch_cpy(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = getenv("GGML_MUSA_CPY_BATCH") == nullptr || atoi(getenv("GGML_MUSA_CPY_BATCH")) != 0;
    if (!enabled || !cpyb_eligible(cgraph->nodes[i])) {
        return 0;
    }
    int idx[CPYB_MAX];
    int n = 0, last = i;
    idx[n++] = i;
    for (int j = i + 1; j < cgraph->n_nodes && n < CPYB_MAX; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (cpyb_is_noop(t)) {
            continue;
        }
        if (!cpyb_eligible(t)) {
            break;
        }
        // independent copies only: no source or destination may overlap an earlier destination in the batch
        bool dep = false;
        for (int k = 0; k < n && !dep; ++k) {
            const ggml_tensor * p = cgraph->nodes[idx[k]];
            dep = cpyb_overlap(t->src[0], p->src[1]) || cpyb_overlap(t->src[1], p->src[1]) || cpyb_overlap(t->src[1], p->src[0]);
        }
        if (dep) {
            break;
        }
        idx[n++] = j;
        last = j;
    }
    if (n < 2) {
        return 0;
    }
    cpyb_args a;
    int64_t nmax = 0;
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * t = cgraph->nodes[idx[k]];
        const ggml_tensor * s = t->src[0], * d = t->src[1];
        cpyb_desc & c = a.d[k];
        c.src = (const char *) s->data;
        c.dst = (char *) d->data;
        // iterate in the source's logical shape; map the flat index to the destination's own shape
        // (ggml_cpy only requires equal element counts) -> require identical shapes here
        for (int q = 0; q < 4; ++q) {
            if (s->ne[q] != d->ne[q]) {
                return 0;
            }
            c.ne[q]  = s->ne[q];
            c.nbs[q] = (int64_t) s->nb[q];
            c.nbd[q] = (int64_t) d->nb[q];
        }
        c.n = ggml_nelements(s);
        nmax = std::max(nmax, c.n);
    }
    const int bx = (int) std::min<int64_t>((nmax + CPYB_BLOCK - 1)/CPYB_BLOCK, 256);
    cpyb_kernel<<<dim3(bx, n), CPYB_BLOCK, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
    return last - i;
}
