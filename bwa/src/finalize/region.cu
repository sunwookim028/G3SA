/*
 * region.cu -- Region filtering, merging, and sorting kernels.
 *
 * Kernels:
 *   patch_regions             -- port of bwa-mem2's mem_sort_dedup_patch / mem_patch_reg
 *   filter_contained_regions  -- post-extension seed containment check
 *   filter_and_score_threshold -- is_alt flagging + score<T compaction
 *   sort_regions              -- full port of bwa-mem2's mem_mark_primary_se (two-pass)
 */

/*
 * Portions of this file are adapted from minhhpham/bwa
 * (https://github.com/minhhpham/bwa), Copyright (c) minhhpham,
 * GPL-3.0 licensed — a GPU port of BWA-MEM (Copyright (c) Dana-Farber
 * Cancer Institute, Broad Institute, Genome Research Ltd.). This file is
 * part of G3SA and is licensed under GPL-3.0 (see LICENSE).
 *
 * mem_patch_reg_gpu (the serial global-alignment merge core, matching
 * bwa-mem2's mem_patch_reg) and patch_regions_serial_fallback (the serial
 * mem_sort_dedup_patch port used for over-budget reads) are near-identical
 * serial ports of minhhpham/bwa's bwamem_GPU.cu logic.  The parallel
 * block-cooperative kernels built on top of that serial core — patch_regions
 * (with its Phase-A/B split), filter_contained_regions, sort_regions, and
 * their block_mark_primary_core / block_sort_* / block_compact helpers — are
 * this project's own original GPU parallelization work.
 */

#include "gpu_types.h"
#include "gmem_alloc.cuh"
#include "bntseq.h"
#include <string.h>
#include <limits.h>
#include "cuda_wrapper.h"
#include "macro.h"
#include "ksw.cuh"

#include "region.cuh"   /* includes cub/cub.cuh, SortRegionsSM, BlockRegionSort */

/* ------------------------------------------------------------------ */
/*  Device-side hash_64 (Thomas Wang hash, matches bwa-mem2 utils.h)  */
/* ------------------------------------------------------------------ */
__device__ static inline uint64_t d_hash_64(uint64_t key)
{
    key += ~(key << 32);
    key ^= (key >> 22);
    key += ~(key << 13);
    key ^= (key >> 8);
    key += (key << 3);
    key ^= (key >> 15);
    key += ~(key << 27);
    key ^= (key >> 31);
    return key;
}

/* ------------------------------------------------------------------ */
/*  Sort comparators matching bwa-mem2:                                */
/*    REG_HLT  (pass 1 / alnreg_hlt):  score DESC->is_alt ASC->hash ASC  */
/*    REG_HLT2 (pass 2 / alnreg_hlt2): is_alt ASC->score DESC->hash ASC  */
/* ------------------------------------------------------------------ */
#define REG_HLT(a, b)  ((a).score > (b).score || \
    ((a).score == (b).score && ((a).is_alt < (b).is_alt || \
    ((a).is_alt == (b).is_alt && (a).hash < (b).hash))))

#define REG_HLT2(a, b) ((a).is_alt < (b).is_alt || \
    ((a).is_alt == (b).is_alt && ((a).score > (b).score || \
    ((a).score == (b).score && (a).hash < (b).hash))))

/* ------------------------------------------------------------------ */
/*  Block-parallel sort helpers                                        */
/*                                                                    */
/*  SortRegionsSM, BlockRegionSort, SORT_REGIONS_BLOCK/ITEMS defined   */
/*  in region.cuh (needed at the launch site in bwamem.cu).            */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*  REG_KEY_HLT: pack (score DESC, is_alt ASC, hash ASC) into uint64_t */
/*  for CUB ascending sort.  16-bit score field covers reads up to     */
/*  65535 bp with match score 1 (bwa-mem2 default a=1).               */
/* ------------------------------------------------------------------ */
__device__ static inline uint64_t reg_key_hlt(const mem_alnreg_t *r)
{
    return  ((uint64_t)(0xFFFFu - (unsigned)(r->score & 0xFFFF)) << 48)
          | ((uint64_t)(r->is_alt & 1u) << 47)
          | (r->hash >> 17);
}

/* REG_KEY_HLT2: pack (is_alt ASC, score DESC, hash ASC) */
__device__ static inline uint64_t reg_key_hlt2(const mem_alnreg_t *r)
{
    return  ((uint64_t)(r->is_alt & 1u) << 63)
          | ((uint64_t)(0xFFFFu - (unsigned)(r->score & 0xFFFF)) << 47)
          | (r->hash >> 17);
}

/* ------------------------------------------------------------------ */
/*  block_sort_hlt / block_sort_hlt2: parallel sort of a[0..n-1].     */
/*  Uses CUB BlockRadixSort on a packed uint64_t key.                  */
/*  After return: sm->new_i[new_rank] = old_rank, a[] in sorted order. */
/*  new_a: caller-supplied temp array (device pool, size n).           */
/* ------------------------------------------------------------------ */
__device__ static void block_sort_hlt(
        mem_alnreg_t *a, int n,
        SortRegionsSM *sm,
        mem_alnreg_t *new_a)
{
    const int tid = threadIdx.x;
    uint64_t thread_keys[SORT_REGIONS_ITEMS];
    int      thread_vals[SORT_REGIONS_ITEMS];
    for (int k = 0; k < SORT_REGIONS_ITEMS; k++) {
        int idx = tid * SORT_REGIONS_ITEMS + k;
        if (idx < n) {
            thread_keys[k] = reg_key_hlt(&a[idx]);
            thread_vals[k] = idx;
        } else {
            thread_keys[k] = UINT64_MAX;
            thread_vals[k] = idx;
        }
    }
    __syncthreads();
    BlockRegionSort(sm->sort_tmp).Sort(thread_keys, thread_vals);
    __syncthreads();

    /* Write sorted indices into SM */
    for (int k = 0; k < SORT_REGIONS_ITEMS; k++) {
        int new_rank = tid * SORT_REGIONS_ITEMS + k;
        if (new_rank < n)
            sm->new_i[new_rank] = (uint16_t)thread_vals[k];
    }
    __syncthreads();

    /* Scatter into new_a, copy back */
    int n_iter = (n + SORT_REGIONS_BLOCK - 1) / SORT_REGIONS_BLOCK;
    for (int k = 0; k < n_iter; k++) {
        int nr = k * SORT_REGIONS_BLOCK + tid;
        if (nr < n) new_a[nr] = a[sm->new_i[nr]];
    }
    __syncthreads();
    for (int k = 0; k < n_iter; k++) {
        int i = k * SORT_REGIONS_BLOCK + tid;
        if (i < n) a[i] = new_a[i];
    }
    __syncthreads();
}

__device__ static void block_sort_hlt2(
        mem_alnreg_t *a, int n,
        SortRegionsSM *sm,
        mem_alnreg_t *new_a)
{
    const int tid = threadIdx.x;
    uint64_t thread_keys[SORT_REGIONS_ITEMS];
    int      thread_vals[SORT_REGIONS_ITEMS];
    for (int k = 0; k < SORT_REGIONS_ITEMS; k++) {
        int idx = tid * SORT_REGIONS_ITEMS + k;
        if (idx < n) {
            thread_keys[k] = reg_key_hlt2(&a[idx]);
            thread_vals[k] = idx;
        } else {
            thread_keys[k] = UINT64_MAX;
            thread_vals[k] = idx;
        }
    }
    __syncthreads();
    BlockRegionSort(sm->sort_tmp).Sort(thread_keys, thread_vals);
    __syncthreads();

    for (int k = 0; k < SORT_REGIONS_ITEMS; k++) {
        int new_rank = tid * SORT_REGIONS_ITEMS + k;
        if (new_rank < n)
            sm->new_i[new_rank] = (uint16_t)thread_vals[k];
    }
    __syncthreads();

    int n_iter = (n + SORT_REGIONS_BLOCK - 1) / SORT_REGIONS_BLOCK;
    for (int k = 0; k < n_iter; k++) {
        int nr = k * SORT_REGIONS_BLOCK + tid;
        if (nr < n) new_a[nr] = a[sm->new_i[nr]];
    }
    __syncthreads();
    for (int k = 0; k < n_iter; k++) {
        int i = k * SORT_REGIONS_BLOCK + tid;
        if (i < n) a[i] = new_a[i];
    }
    __syncthreads();
}

/* ------------------------------------------------------------------ */
/*  block_mark_primary_core: block-cooperative port of                  */
/*  bwa-mem2 mem_mark_primary_se_core.                                  */
/*                                                                    */
/*  Outer loop i is block-serial (preserves bwa-mem2 ordering).        */
/*  Inner loop j is parallelized: thread t owns j in {t, t+BLOCK, …}. */
/*  Block-min reduction finds first (smallest j) overlapping primary.  */
/*  The owning thread writes a[j*].sub / a[j*].sub_n; thread 0 writes  */
/*  a[i].secondary = j*.                                               */
/*  __syncthreads() at end of each outer iter ensures coherence.       */
/* ------------------------------------------------------------------ */
__device__ static void block_mark_primary_core(
        const mem_opt_t *opt, int n, mem_alnreg_t *a,
        int *reduce_SM)
{
    const int tid    = threadIdx.x;
    const int warpid = tid >> 5;
    const int lane   = tid & 31;
    const int n_warps = SORT_REGIONS_BLOCK >> 5;  /* 8 warps */

    int gap_tmp = opt->a + opt->b;
    gap_tmp = opt->o_del + opt->e_del > gap_tmp ? opt->o_del + opt->e_del : gap_tmp;
    gap_tmp = opt->o_ins + opt->e_ins > gap_tmp ? opt->o_ins + opt->e_ins : gap_tmp;
    const float mask_level = opt->mask_level;

    for (int i = 1; i < n; i++) {
        /* Cache outer constants so all threads read once */
        const int ai_qb    = a[i].qb;
        const int ai_qe    = a[i].qe;
        const int ai_score = a[i].score;
        const int ai_isalt = a[i].is_alt;
        const int ai_len   = ai_qe - ai_qb;

        /* Each thread scans its owned j-values; track local minimum j
         * where the overlap predicate fires (i.e., first_j per thread). */
        int local_min_j = i;  /* sentinel: i means "no candidate found" */

        for (int k = 0; ; k++) {
            int j = tid + k * SORT_REGIONS_BLOCK;
            if (j >= i) break;
            if (a[j].secondary >= 0) continue;
            int b_max = a[j].qb > ai_qb ? a[j].qb : ai_qb;
            int e_min = a[j].qe < ai_qe ? a[j].qe : ai_qe;
            if (e_min > b_max) {
                int min_l = ai_len < (a[j].qe - a[j].qb) ? ai_len : (a[j].qe - a[j].qb);
                if (e_min - b_max >= (int)(min_l * mask_level)) {
                    local_min_j = j;
                    break;  /* thread owns j in stride order; first hit is smallest j */
                }
            }
        }

        /* Block-min reduction to find the globally-smallest j* */
        int wv = local_min_j;
        for (int m = 16; m > 0; m >>= 1)
            wv = min(wv, __shfl_xor_sync(0xffffffff, wv, m));
        if (lane == 0) reduce_SM[warpid] = wv;
        __syncthreads();
        if (warpid == 0) {
            wv = (lane < n_warps) ? reduce_SM[lane] : i;
            for (int m = 4; m > 0; m >>= 1)
                wv = min(wv, __shfl_xor_sync(0xffffffff, wv, m));
            if (lane == 0) reduce_SM[0] = wv;
        }
        __syncthreads();
        const int jstar = reduce_SM[0];

        if (jstar < i) {
            /* Thread 0 writes a[i].secondary = jstar */
            if (tid == 0)
                a[i].secondary = jstar;
            /* The thread that owns jstar writes a[jstar].sub and .sub_n */
            if (jstar % SORT_REGIONS_BLOCK == tid) {
                if (a[jstar].sub == 0) a[jstar].sub = ai_score;
                if (a[jstar].score - ai_score <= gap_tmp &&
                    (a[jstar].is_alt || !ai_isalt))
                    ++a[jstar].sub_n;
            }
        }
        __syncthreads();
    }
}

/* ------------------------------------------------------------------ */
/*  Constants matching bwa-mem2's mem_patch_reg thresholds             */
/* ------------------------------------------------------------------ */
#define PATCH_MAX_R_BW      0.05f
#define PATCH_MIN_SC_RATIO  0.90f

/* ------------------------------------------------------------------ */
/*  Device helper: mem_patch_reg_gpu                                   */
/*                                                                    */
/*  Attempt to merge two colinear regions via global alignment.       */
/*  Returns alignment score > 0 on success, 0 on failure.             */
/*  *_w receives the bandwidth used.                                  */
/*                                                                    */
/*  Precondition: a->rid == b->rid && a->rb <= b->rb                 */
/*  Matches bwa-mem2/src/bwamem.cpp mem_patch_reg exactly.            */
/*                                                                    */
/*  Reverse-strand path note: this function reverses the query buffer */
/*  IN-PLACE when a->rb >= l_pac, then restores it.  Concurrent calls */
/*  from multiple threads on the SAME read's query buffer would race, */
/*  so only thread 0 (the Phase B serial fallback) may call this      */
/*  routine.                                                          */
/* ------------------------------------------------------------------ */
__device__ static int mem_patch_reg_gpu(
        const mem_opt_t *opt,
        const bntseq_t *bns,
        const uint8_t  *pac,
        uint8_t        *query,   /* full read sequence (may be temporarily reversed) */
        const mem_alnreg_t *a,
        const mem_alnreg_t *b,
        int *_w,
        void *d_buffer_ptr)
{
    int w, score, q_s, r_s;
    double r;
    int64_t l_pac = bns->l_pac;

    /* a and b must not bridge the forward-reverse boundary */
    if (a->rb < l_pac && b->rb >= l_pac) return 0;

    /* colinearity check: a must precede b in both query and ref */
    if (a->qb >= b->qb || a->qe >= b->qe || a->re >= b->re) return 0;

    /* bandwidth and ratio filter */
    w = (int)((a->re - b->rb) - (a->qe - b->qb));
    w = w > 0 ? w : -w;
    r = (double)(a->re - b->rb) / (double)(b->re - a->rb)
      - (double)(a->qe - b->qb) / (double)(b->qe - a->qb);
    r = r > 0.0 ? r : -r;

    if (a->re < b->rb || a->qe < b->qb) {
        /* gap between the two regions */
        if (w > (opt->w << 1) || r >= PATCH_MAX_R_BW) return 0;
    } else {
        /* overlapping */
        if (w > (opt->w << 2) || r >= PATCH_MAX_R_BW * 2) return 0;
    }

    w += a->w + b->w;
    w = w < (opt->w << 2) ? w : (opt->w << 2);

    /* ---- global alignment (score only, no CIGAR) ---- */
    int l_query_patch = b->qe - a->qb;
    int64_t rb = a->rb, re = b->re;

    if (l_query_patch <= 0 || rb >= re || (rb < l_pac && re > l_pac))
        return 0;

    /* Pre-filter — cheap upper-bound check before the expensive alignment.
     * Skip bns_get_seq_gpu + query reversal + ksw_global2 when the best-case
     * score (perfect match for every base) cannot satisfy PATCH_MIN_SC_RATIO.
     * Safe: max_score >= actual_score, so returning 0 here is always correct. */
    {
        int q_s_pre = (int)((double)(b->qe - a->qb) / ((b->qe - b->qb) + (a->qe - a->qb))
                            * (b->score + a->score) + 0.499);
        int r_s_pre = (int)((double)(b->re - a->rb) / ((b->re - b->rb) + (a->re - a->rb))
                            * (b->score + a->score) + 0.499);
        int denom = q_s_pre > r_s_pre ? q_s_pre : r_s_pre;
        int max_score = (l_query_patch < (int)(re - rb) ? l_query_patch : (int)(re - rb))
                        * opt->mat[0];
        if (denom > 0 && (double)max_score / denom < PATCH_MIN_SC_RATIO)
            return 0;
    }

    /* fetch reference sequence */
    int64_t rlen;
    uint8_t *rseq = bns_get_seq_gpu(l_pac, pac, rb, re, &rlen, d_buffer_ptr);
    if (re - rb != rlen) return 0;

    uint8_t *qseq = query + a->qb;

    /* if on reverse strand, reverse both query and reference */
    if (rb >= l_pac) {
        for (int i = 0; i < l_query_patch >> 1; ++i) {
            uint8_t tmp = qseq[i];
            qseq[i] = qseq[l_query_patch - 1 - i];
            qseq[l_query_patch - 1 - i] = tmp;
        }
        for (int i = 0; i < (int)rlen >> 1; ++i) {
            uint8_t tmp = rseq[i];
            rseq[i] = rseq[rlen - 1 - i];
            rseq[rlen - 1 - i] = tmp;
        }
    }

    if (l_query_patch == (int)(re - rb) && w == 0) {
        /* no gap -- simple scoring */
        score = 0;
        for (int i = 0; i < l_query_patch; ++i)
            score += opt->mat[rseq[i] * 5 + qseq[i]];
    } else {
        /* compute bandwidth */
        int max_ins = (int)((double)(((l_query_patch + 1) >> 1) * opt->mat[0] - opt->o_ins) / opt->e_ins + 1.);
        int max_del = (int)((double)(((l_query_patch + 1) >> 1) * opt->mat[0] - opt->o_del) / opt->e_del + 1.);
        int max_gap = max_ins > max_del ? max_ins : max_del;
        max_gap = max_gap > 1 ? max_gap : 1;
        int w2 = (max_gap + abs((int)rlen - l_query_patch) + 1) >> 1;
        w2 = w2 < w ? w2 : w;
        int min_w = abs((int)rlen - l_query_patch) + 3;
        w2 = w2 > min_w ? w2 : min_w;

        /* NW alignment -- score only (n_cigar=NULL skips backtrack) */
        score = ksw_global2(l_query_patch, qseq, (int)rlen, rseq, 5, opt->mat,
                            opt->o_del, opt->e_del, opt->o_ins, opt->e_ins,
                            w2, NULL, NULL, d_buffer_ptr);
    }

    /* reverse query back if we reversed it */
    if (rb >= l_pac) {
        for (int i = 0; i < l_query_patch >> 1; ++i) {
            uint8_t tmp = qseq[i];
            qseq[i] = qseq[l_query_patch - 1 - i];
            qseq[l_query_patch - 1 - i] = tmp;
        }
    }

    /* score quality check */
    q_s = (int)((double)(b->qe - a->qb) / ((b->qe - b->qb) + (a->qe - a->qb))
                * (b->score + a->score) + 0.499);
    r_s = (int)((double)(b->re - a->rb) / ((b->re - b->rb) + (a->re - a->rb))
                * (b->score + a->score) + 0.499);

    if ((double)score / (q_s > r_s ? q_s : r_s) < PATCH_MIN_SC_RATIO)
        return 0;

    *_w = w;
    return score;
}

/* ------------------------------------------------------------------ */
/*  patch_regions parallel sort key packers + block sort helpers.       */
/*                                                                      */
/*  Two sorts in patch_regions:                                         */
/*    Step 1 (alnreg_slt2): sort by re ASC, then orig-index ASC (stable). */
/*    Step 5 (alnreg_slt):  sort by score DESC, rb ASC, qb ASC.         */
/*                                                                      */
/*  Tie-break by original index keeps the parallel sort equivalent to    */
/*  the serial insertion sort (which is stable).  MAX_N_ALN=3072 fits in */
/*  12 bits.                                                             */
/*                                                                      */
/*  Used only when n >= SORT_PARALLEL_MIN (default 64).  Below the       */
/*  threshold, thread 0 runs an in-place insertion sort, which avoids    */
/*  the CUB launch / new_a allocator overhead and matches bwa-mem2's     */
/*  stable sort exactly.                                                  */
/* ------------------------------------------------------------------ */

/* Pack: 48 bits re ASC | 4 bits zero | 12 bits idx ASC.
 * re for hg38 (~6.2e9) fits in 33 bits → 48-bit field is ample.            */
__device__ static inline uint64_t reg_key_re(const mem_alnreg_t *r, int idx)
{
    return  ((uint64_t)(r->re & 0x0000FFFFFFFFFFFFull) << 16)
          | ((uint32_t)idx & 0x0FFFu);
}

/* Pack: 16 bits score DESC | 36 bits rb ASC | 12 bits qb ASC.
 * rb (hg38 ~6.2e9) fits in 33 bits → 36-bit field is ample.
 * qb (read length) fits in 12 bits for reads up to 4095 bp.                */
__device__ static inline uint64_t reg_key_score(const mem_alnreg_t *r)
{
    /* score is int; serial uses signed compare a[j].score < i_score.
     * Clamp to [0, 0xFFFF] for packing; bwa-mem2 scores are non-negative. */
    unsigned s = (r->score < 0) ? 0u
                : ((r->score > 0xFFFF) ? 0xFFFFu : (unsigned)r->score);
    return  ((uint64_t)(0xFFFFu - s) << 48)
          | ((uint64_t)(r->rb & 0x0000000FFFFFFFFFull) << 12)
          | ((uint32_t)r->qb & 0x0FFFu);
}

/* block_sort_by_re: parallel stable sort of a[0..n-1] by re ASC, idx ASC.
 * new_a is a caller-supplied device-pool temp array, size n.               */
__device__ static void block_sort_by_re(
        mem_alnreg_t *a, int n,
        SortRegionsSM *sm,
        mem_alnreg_t *new_a)
{
    const int tid = threadIdx.x;
    uint64_t thread_keys[SORT_REGIONS_ITEMS];
    int      thread_vals[SORT_REGIONS_ITEMS];
    for (int k = 0; k < SORT_REGIONS_ITEMS; k++) {
        int idx = tid * SORT_REGIONS_ITEMS + k;
        if (idx < n) {
            thread_keys[k] = reg_key_re(&a[idx], idx);
            thread_vals[k] = idx;
        } else {
            thread_keys[k] = UINT64_MAX;
            thread_vals[k] = idx;
        }
    }
    __syncthreads();
    BlockRegionSort(sm->sort_tmp).Sort(thread_keys, thread_vals);
    __syncthreads();

    for (int k = 0; k < SORT_REGIONS_ITEMS; k++) {
        int new_rank = tid * SORT_REGIONS_ITEMS + k;
        if (new_rank < n)
            sm->new_i[new_rank] = (uint16_t)thread_vals[k];
    }
    __syncthreads();

    int n_iter = (n + SORT_REGIONS_BLOCK - 1) / SORT_REGIONS_BLOCK;
    for (int k = 0; k < n_iter; k++) {
        int nr = k * SORT_REGIONS_BLOCK + tid;
        if (nr < n) new_a[nr] = a[sm->new_i[nr]];
    }
    __syncthreads();
    for (int k = 0; k < n_iter; k++) {
        int i = k * SORT_REGIONS_BLOCK + tid;
        if (i < n) a[i] = new_a[i];
    }
    __syncthreads();
}

/* block_sort_by_score: parallel sort of a[0..n-1] by score DESC, rb ASC, qb ASC. */
__device__ static void block_sort_by_score(
        mem_alnreg_t *a, int n,
        SortRegionsSM *sm,
        mem_alnreg_t *new_a)
{
    const int tid = threadIdx.x;
    uint64_t thread_keys[SORT_REGIONS_ITEMS];
    int      thread_vals[SORT_REGIONS_ITEMS];
    for (int k = 0; k < SORT_REGIONS_ITEMS; k++) {
        int idx = tid * SORT_REGIONS_ITEMS + k;
        if (idx < n) {
            thread_keys[k] = reg_key_score(&a[idx]);
            thread_vals[k] = idx;
        } else {
            thread_keys[k] = UINT64_MAX;
            thread_vals[k] = idx;
        }
    }
    __syncthreads();
    BlockRegionSort(sm->sort_tmp).Sort(thread_keys, thread_vals);
    __syncthreads();

    for (int k = 0; k < SORT_REGIONS_ITEMS; k++) {
        int new_rank = tid * SORT_REGIONS_ITEMS + k;
        if (new_rank < n)
            sm->new_i[new_rank] = (uint16_t)thread_vals[k];
    }
    __syncthreads();

    int n_iter = (n + SORT_REGIONS_BLOCK - 1) / SORT_REGIONS_BLOCK;
    for (int k = 0; k < n_iter; k++) {
        int nr = k * SORT_REGIONS_BLOCK + tid;
        if (nr < n) new_a[nr] = a[sm->new_i[nr]];
    }
    __syncthreads();
    for (int k = 0; k < n_iter; k++) {
        int i = k * SORT_REGIONS_BLOCK + tid;
        if (i < n) a[i] = new_a[i];
    }
    __syncthreads();
}

/* ------------------------------------------------------------------ */
/*  block_compact: parallel compaction helper                           */
/*                                                                      */
/*  Compacts a[0..n-1] in-place, keeping only entries where            */
/*  a[i].qe > a[i].qb.  Uses CUB BlockScan<int,256> exclusive-sum     */
/*  on SORT_REGIONS_ITEMS flags per thread (blocked arrangement,       */
/*  tile = tid * SORT_REGIONS_ITEMS .. tid * SORT_REGIONS_ITEMS + 11). */
/*                                                                      */
/*  new_a: caller-supplied device-pool temp buffer of size >= n.       */
/*  scan_tmp: caller-supplied BlockCompact::TempStorage (shared).      */
/*  Returns: new count m.                                               */
/*                                                                      */
/*  Requires __syncthreads() before return; caller does NOT need one   */
/*  after.                                                              */
/* ------------------------------------------------------------------ */
__device__ static int block_compact(
        mem_alnreg_t *a, int n,
        mem_alnreg_t *new_a,
        typename BlockCompact::TempStorage &scan_tmp)
{
    const int tid = threadIdx.x;

    /* Each thread loads flags for its tile [tid*12, tid*12+12). */
    int flags[SORT_REGIONS_ITEMS];
    for (int k = 0; k < SORT_REGIONS_ITEMS; k++) {
        int i = tid * SORT_REGIONS_ITEMS + k;
        flags[k] = (i < n && a[i].qe > a[i].qb) ? 1 : 0;
    }

    /* ExclusiveSum: prefix[k] = destination index for element tid*12+k. */
    int prefix[SORT_REGIONS_ITEMS];
    int block_total;
    BlockCompact(scan_tmp).ExclusiveSum(flags, prefix, block_total);
    /* Note: BlockCompact::ExclusiveSum includes an implicit __syncthreads()
     * inside CUB.  We add one after to ensure new_a writes are visible. */

    /* Scatter live entries to new_a. */
    for (int k = 0; k < SORT_REGIONS_ITEMS; k++) {
        int i = tid * SORT_REGIONS_ITEMS + k;
        if (i < n && flags[k])
            new_a[prefix[k]] = a[i];
    }
    __syncthreads();

    /* Copy compacted result back to a (only new block_total entries). */
    int n_new = block_total;
    int n_iter = (n_new + SORT_REGIONS_BLOCK - 1) / SORT_REGIONS_BLOCK;
    for (int k = 0; k < n_iter; k++) {
        int i = k * SORT_REGIONS_BLOCK + tid;
        if (i < n_new) a[i] = new_a[i];
    }
    __syncthreads();

    return n_new;
}

/* ------------------------------------------------------------------ */
/*  patch_regions kernel (block-cooperative parallel dedup+merge)       */
/*                                                                      */
/*  GPU port of bwa-mem2's mem_sort_dedup_patch.  For each read:        */
/*    1. Sort by re ASC: parallel CUB radix sort for n >= 64,            */
/*       serial insertion sort otherwise (CUB launch overhead).         */
/*    2. Parallel init n_comp = 1 (stride).                              */
/*    3. Pairwise dedup / merge (block-cooperative):                     */
/*         - Outer i loop is block-serial (preserves bwa-mem2 order).    */
/*         - Phase A: parallel scan of j ∈ [j_min, i-1].  Each thread t  */
/*           owns j ∈ {t, t+BLOCK, …}.  Read-only on a[] (uses cached    */
/*           a[i] snapshot in registers).  Writes per-j classification   */
/*           code to sm->class_code[j]: high-overlap dedup pairs get      */
/*           killed immediately (cc==1/2); all other (merge-candidate)   */
/*           pairs are left at cc==0 (idle) and deferred to Phase B.      */
/*         - Phase B: thread 0 walks j = i-1 .. 0 in bwa-mem2 order,     */
/*           applying the dedup writes from class_code[].  Merge         */
/*           candidates (cc==0, not already dead) are only attempted     */
/*           once any_merge has already been set true earlier in the     */
/*           same outer iteration (post-any_merge serial fallback via    */
/*           mem_patch_reg_gpu) — this matches the empirical result that */
/*           initial-pass merge attempts (both forward- and reverse-      */
/*           strand) essentially never succeed on tested datasets, so    */
/*           skipping them outright is a no-op simplification.            */
/*    4. Compact dead entries: CUB BlockScan for n >= 64,                 */
/*       serial thread-0 fallback for small n.                            */
/*    5. Sort by score DESC, rb ASC, qb ASC: parallel CUB for n >= 64.   */
/*    6. Parallel mark exact duplicates + CUB BlockScan compact (same).  */
/*                                                                      */
/*  Equivalence with serial bwa-mem2:                                    */
/*    - Phase A reads a[] only; per-j writes go to disjoint sm[j].       */
/*    - Phase B applies writes in the exact bwa-mem2 j-order in thread 0.*/
/*    - any_merge fallback re-runs full mem_patch_reg_gpu for remaining  */
/*      j's, so the chained-merge mutation semantics are preserved.      */
/*                                                                      */
/*  Launch: <<<batch_size, SORT_REGIONS_BLOCK, PATCH_REGIONS_SM_BYTES>>>. */
/* ------------------------------------------------------------------ */

/* class_code values stored in sm->class_code[j] by Phase A: */
#define PATCH_CC_IDLE         0   /* dead / predicate failed / no overlap / deferred merge candidate */
#define PATCH_CC_DEDUP_KILLI  1   /* high-overlap, p->score <  q->score   */
#define PATCH_CC_DEDUP_KILLJ  2   /* high-overlap, p->score >= q->score   */

/* Fallback for n > MAX_N_ALN: run the entire bwa-mem2 mem_sort_dedup_patch
 * sequentially in thread 0.  Used for the rare centromeric reads where the
 * region count exceeds our parallel SM budget.                            */
__device__ static void patch_regions_serial_fallback(
        const mem_opt_t *d_opt,
        const bntseq_t  *d_bns,
        const uint8_t   *d_pac,
        uint8_t         *query,
        mem_alnreg_v    *regs,
        void            *d_buffer_ptr)
{
    int n = regs->n;
    if (n <= 1) return;
    mem_alnreg_t *a = regs->a;
    const int    max_chain_gap    = d_opt->max_chain_gap;
    const float  mask_level_redun = d_opt->mask_level_redun;

    /* 1: insertion sort by re ASC */
    {
        mem_alnreg_t tmp;
        for (int i = 1; i < n; i++) {
            int64_t i_re = a[i].re;
            int j = i - 1;
            if (a[j].re > i_re) {
                tmp = a[i];
                do { a[j + 1] = a[j]; j--; }
                while (j >= 0 && a[j].re > i_re);
                a[j + 1] = tmp;
            }
        }
    }
    /* 2: init n_comp */
    for (int i = 0; i < n; i++) a[i].n_comp = 1;
    /* 3: dedup / merge */
    for (int i = 1; i < n; i++) {
        mem_alnreg_t *p = &a[i];
        if (p->qe == p->qb) continue;
        for (int j = i - 1; j >= 0
                && p->rid == a[j].rid
                && p->rb  <  a[j].re + max_chain_gap; --j) {
            mem_alnreg_t *q = &a[j];
            if (q->qe == q->qb) continue;
            int64_t or_ = q->re - p->rb;
            int oq = q->qb < p->qb ? q->qe - p->qb : p->qe - q->qb;
            int mr = (int)(q->re - q->rb < p->re - p->rb
                            ? q->re - q->rb : p->re - p->rb);
            int mq = (q->qe - q->qb < p->qe - p->qb)
                            ? q->qe - q->qb : p->qe - p->qb;
            if (or_ > (int64_t)(mask_level_redun * mr)
                && oq > (int)(mask_level_redun * mq)) {
                if (p->score < q->score) { p->qe = p->qb; break; }
                else q->qe = q->qb;
            } else if (q->rb < p->rb) {
                int w;
                int sc = mem_patch_reg_gpu(d_opt, d_bns, d_pac, query,
                                          q, p, &w, d_buffer_ptr);
                if (sc > 0) {
                    p->n_comp += q->n_comp + 1;
                    if (q->seedcov > p->seedcov) p->seedcov = q->seedcov;
                    if (q->sub > p->sub) p->sub = q->sub;
                    if (q->csub > p->csub) p->csub = q->csub;
                    p->qb = q->qb;
                    p->rb = q->rb;
                    p->truesc = sc; p->score = sc; p->w = w;
                    q->qb = q->qe;
                }
            }
        }
    }
    /* 4: compact dead */
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (a[i].qe > a[i].qb) {
            if (m != i) a[m] = a[i];
            m++;
        }
    }
    n = m;
    if (n > 1) {
        /* 5: sort by score DESC, rb ASC, qb ASC */
        mem_alnreg_t tmp;
        for (int i = 1; i < n; i++) {
            int     i_score = a[i].score;
            int64_t i_rb    = a[i].rb;
            int     i_qb    = a[i].qb;
            int j = i - 1;
            if (a[j].score < i_score
                || (a[j].score == i_score
                    && (a[j].rb > i_rb
                        || (a[j].rb == i_rb && a[j].qb > i_qb)))) {
                tmp = a[i];
                do { a[j + 1] = a[j]; j--; }
                while (j >= 0
                    && (a[j].score < i_score
                        || (a[j].score == i_score
                            && (a[j].rb > i_rb
                                || (a[j].rb == i_rb && a[j].qb > i_qb)))));
                a[j + 1] = tmp;
            }
        }
        /* 6: exact-dup mark + compact */
        for (int i = 1; i < n; i++) {
            if (a[i].score == a[i - 1].score
                && a[i].rb    == a[i - 1].rb
                && a[i].qb    == a[i - 1].qb)
                a[i].qe = a[i].qb;
        }
        m = 0;
        for (int i = 0; i < n; i++) {
            if (a[i].qe > a[i].qb) {
                if (m != i) a[m] = a[i];
                m++;
            }
        }
        n = m;
    }
    regs->n = n;
}

__launch_bounds__(256, 3)
__global__ void patch_regions(
        const mem_opt_t  *d_opt,
        const bntseq_t   *d_bns,
        const uint8_t    *d_pac,
        uint8_t          *d_seq,        /* packed read sequences */
        const int        *d_seq_offset,
        mem_alnreg_v     *d_regs,
        void             *d_buffer_pools,
        int               batch_size)
{
#if __CUDACC_VER_MAJOR__ < 13
    asm volatile (".pragma \"enable_smem_spilling\";");
#endif
    int seqID = blockIdx.x;
    if (seqID >= batch_size) return;
    int n = d_regs[seqID].n;
    if (n <= 1) return;

    const int tid = threadIdx.x;

    mem_alnreg_t *a = d_regs[seqID].a;

    /* Pool 32-63 carries ksw_global2 / bns_get_seq_gpu allocations.
     * CUDAKernelMalloc is atomic on the pool offset → safe for parallel
     * calls from multiple threads of the same block.                       */
    void *d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, 32 + seqID % 32);

    /* get this read's sequence */
    int seq_off  = d_seq_offset[seqID];
    uint8_t *query = d_seq + seq_off;

    /* Rare fallback: n > MAX_N_ALN exceeds our SM scratch budget.  Run the
     * entire serial bwa-mem2 algorithm in thread 0.  These centromeric reads
     * are rare (<0.01% of total) so the perf cost is negligible.            */
    if (n > MAX_N_ALN) {
        if (tid == 0)
            patch_regions_serial_fallback(d_opt, d_bns, d_pac, query,
                                          &d_regs[seqID], d_buffer_ptr);
        return;
    }

    /* Shared memory layout: SortRegionsSM (CUB temp + new_i + reduce_SM)
     * followed by PatchSM (j_min/j_kill_max/has_merge + new_a_ptr +
     * class_code[MAX_N_ALN]).                                             */
    extern __shared__ char _patch_SM[];
    SortRegionsSM *sm  = (SortRegionsSM *)_patch_SM;
    PatchSM       *psm = (PatchSM *)((char *)_patch_SM + sizeof(SortRegionsSM));

    /* Static SM for CUB BlockScan TempStorage (~64 B on sm_86). */
    __shared__ typename BlockCompact::TempStorage compact_scan_tmp;

    /* --- Step 1: sort by re ASC.                                       */
    const int SORT_PARALLEL_MIN = 64;
    if (n >= SORT_PARALLEL_MIN) {
        if (tid == 0) {
            psm->new_a_ptr = CUDAKernelMalloc(
                    d_buffer_ptr, (size_t)n * sizeof(mem_alnreg_t), 8);
        }
        __syncthreads();
        mem_alnreg_t *new_a = (mem_alnreg_t *)psm->new_a_ptr;
        block_sort_by_re(a, n, sm, new_a);
    } else {
        if (tid == 0) {
            mem_alnreg_t tmp;
            for (int i = 1; i < n; i++) {
                int64_t i_re = a[i].re;
                int j = i - 1;
                if (a[j].re > i_re) {
                    tmp = a[i];
                    do {
                        a[j + 1] = a[j];
                        j--;
                    } while (j >= 0 && a[j].re > i_re);
                    a[j + 1] = tmp;
                }
            }
        }
        __syncthreads();
    }

    /* --- Step 2: init n_comp (parallel stride) --- */
    for (int i = tid; i < n; i += SORT_REGIONS_BLOCK)
        a[i].n_comp = 1;
    __syncthreads();

    /* --- Step 3: block-cooperative pairwise dedup / merge --- */
    const int    max_chain_gap   = d_opt->max_chain_gap;
    const float  mask_level_redun = d_opt->mask_level_redun;

    for (int i = 1; i < n; i++) {
        /* Cache outer-iteration constants in registers (Phase-A snapshot). */
        const int     p_rid      = a[i].rid;
        const int64_t p_rb_in    = a[i].rb;
        const int64_t p_re_in    = a[i].re;
        const int     p_qb_in    = a[i].qb;
        const int     p_qe_in    = a[i].qe;
        const int     p_score_in = a[i].score;

        /* Quick gate: i-1 has the largest re among j<i after the sort, so
         * if no overlap with i-1, no overlap with anything.                */
        bool quick_skip = false;
        if (tid == 0) {
            quick_skip = (p_qe_in == p_qb_in)              /* i already dead */
                      || (a[i - 1].rid != p_rid)
                      || (p_rb_in >= a[i - 1].re + max_chain_gap);
            sm->reduce_SM[1] = quick_skip ? 1 : 0;
        }
        __syncthreads();
        if (sm->reduce_SM[1]) continue;

        /* Compute j_min: first j in [0,i) where a[j].re + max_chain_gap > p_rb_in.
         * Binary search on the re-sorted array: O(log i) vs the previous O(i) scan.
         * Postcondition: for j in [j_min,i-1], a[j].re+gap > p_rb_in AND rid==p_rid
         * (same-rid elements are contiguous in the global-coordinate-sorted array,
         * and quick_skip guarantees a[i-1].rid==p_rid). */
        if (tid == 0) {
            int64_t threshold = p_rb_in - (int64_t)max_chain_gap;
            int lo = 0, hi = i;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (a[mid].re <= threshold)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            /* Safety: skip any stray different-rid element (global coords keep
             * same-rid contiguous, so this scan takes at most a couple steps). */
            while (lo < i && a[lo].rid != p_rid) lo++;
            psm->j_min = lo;
        }
        __syncthreads();
        const int j_min = psm->j_min;

        /* --- Phase A: parallel classification of each j ∈ [j_min, i-1]. ---
         * Each thread t owns j ∈ {j_min + t, j_min + t + BLOCK, ...}.        */
        for (int j = j_min + tid; j < i; j += SORT_REGIONS_BLOCK) {
            uint8_t cc = PATCH_CC_IDLE;

            if (a[j].qe == a[j].qb) {
                psm->class_code[j] = cc;
                continue;
            }
            const int64_t q_rb = a[j].rb;
            const int64_t q_re = a[j].re;
            const int     q_qb = a[j].qb;
            const int     q_qe = a[j].qe;
            const int     q_sc = a[j].score;

            int64_t or_ = q_re - p_rb_in;
            int     oq  = (q_qb < p_qb_in) ? (q_qe - p_qb_in) : (p_qe_in - q_qb);
            int     mr  = (int)((q_re - q_rb < p_re_in - p_rb_in)
                                  ? (q_re - q_rb) : (p_re_in - p_rb_in));
            int     mq  = (q_qe - q_qb < p_qe_in - p_qb_in)
                                  ? (q_qe - q_qb) : (p_qe_in - p_qb_in);
            bool high_overlap = (or_ > (int64_t)(mask_level_redun * (float)mr))
                              && (oq > (int)(mask_level_redun * (float)mq));

            if (high_overlap) {
                cc = (p_score_in < q_sc) ? PATCH_CC_DEDUP_KILLI
                                         : PATCH_CC_DEDUP_KILLJ;
            }
            /* else: q_rb < p_rb_in merge candidates (forward or reverse
             * strand) are left at cc = PATCH_CC_IDLE.  The Phase-A merge
             * fast path is disabled by default (forward-strand merges are
             * skipped entirely here; reverse-strand merges have a 0%
             * success rate on all tested datasets — see PatchSM).  These
             * pairs are instead picked up by Phase B's post-any_merge
             * serial fallback, once an earlier merge in the same outer
             * iteration makes any_merge true. */
            psm->class_code[j] = cc;
        }
        __syncthreads();

        /* --- Phase B: thread 0 applies writes in bwa-mem2 j-order. ----
         * Segment 1: j ∈ [j_min, i-1].  Break conditions guaranteed for
         * !any_merge (j>=j_min ↔ a[j].re+gap > p_rb_in and rid==p_rid).
         * After any_merge, p->rb can only decrease so the gap condition is
         * even more satisfied for remaining j in [j_min,i-1] — still no
         * break checks needed.  Check SM class_code FIRST for !any_merge:
         * ~90% of entries are IDLE, saving the a[j] global-mem load.
         * Segment 2: j ∈ [0, j_min-1], only entered if any_merge mutated
         * p->rb enough to extend the window below original j_min.        */
        if (tid == 0) {
            mem_alnreg_t *p = &a[i];
            bool any_merge = false;
            bool killed_i  = false;
            /* --- Segment 1 --- */
            for (int j = i - 1; j >= j_min; --j) {
                mem_alnreg_t *q;
                if (!any_merge) {
                    uint8_t cc = psm->class_code[j];
                    if (cc == PATCH_CC_IDLE) continue;   /* skip a[j] load! */
                    q = &a[j];
                    if (q->qe == q->qb) continue;        /* dead (safety) */
                    if (cc == PATCH_CC_DEDUP_KILLI) {
                        p->qe = p->qb;
                        killed_i = true;
                        break;
                    }
                    if (cc == PATCH_CC_DEDUP_KILLJ) {
                        q->qe = q->qb;
                        continue;
                    }
                    /* All possible cc values (IDLE/KILLI/KILLJ) are handled
                     * above (continue/continue/break), so this point is
                     * unreachable while !any_merge.  The shared serial-
                     * recompute block below is reached only via the
                     * any_merge==true (else) branch, for j's not yet visited
                     * by this walk. */
                } else {
                    q = &a[j];
                    if (q->qe == q->qb) continue;
                }
                /* Serial recompute path: reached only once any_merge is true
                 * (post-any_merge fallback, re-evaluating j's below the merge
                 * point against the mutated p).  j >= j_min: rid and gap
                 * guaranteed (post-merge p->rb decreases → gap only widens). */
                {
                    int64_t or_ = q->re - p->rb;
                    int oq = q->qb < p->qb ? q->qe - p->qb : p->qe - q->qb;
                    int mr = (int)(q->re - q->rb < p->re - p->rb
                                    ? q->re - q->rb : p->re - p->rb);
                    int mq = (q->qe - q->qb < p->qe - p->qb)
                                    ? q->qe - q->qb : p->qe - p->qb;
                    if (or_ > (int64_t)(mask_level_redun * mr)
                        && oq > (int)(mask_level_redun * mq)) {
                        if (p->score < q->score) {
                            p->qe = p->qb;
                            killed_i = true;
                            break;
                        } else {
                            q->qe = q->qb;
                        }
                    } else if (q->rb < p->rb) {
                        /* Skip initial rev-strand merge attempts (cc==4, !any_merge):
                         * empirically 0% success rate on all tested datasets.
                         * Only attempt the serial merge for the post-any_merge
                         * fallback (any_merge==true), where the pair may have
                         * become forward-strand-compatible after the prior
                         * merge widened the window. */
                        if (any_merge) {
                            int w_out;
                            int score = mem_patch_reg_gpu(d_opt, d_bns, d_pac, query,
                                                          q, p, &w_out, d_buffer_ptr);
                            if (score > 0) {
                                p->n_comp += q->n_comp + 1;
                                if (q->seedcov > p->seedcov) p->seedcov = q->seedcov;
                                if (q->sub     > p->sub)     p->sub     = q->sub;
                                if (q->csub    > p->csub)    p->csub    = q->csub;
                                p->qb = q->qb;
                                p->rb = q->rb;
                                p->truesc = score;
                                p->score  = score;
                                p->w      = w_out;
                                q->qb = q->qe;
                                any_merge = true;
                            }
                        }
                    }
                }
            }
            /* --- Segment 2: j ∈ [0, j_min-1] ---
             * Only needed when any_merge mutated p->rb, possibly widening
             * the overlap window below original j_min.  Full rid/re checks. */
            if (any_merge && !killed_i) {
                for (int j = j_min - 1; j >= 0; --j) {
                    if (p->rid != a[j].rid) break;
                    if (p->rb  >= a[j].re + max_chain_gap) break;
                    mem_alnreg_t *q = &a[j];
                    if (q->qe == q->qb) continue;
                    {
                        int64_t or_ = q->re - p->rb;
                        int oq = q->qb < p->qb ? q->qe - p->qb : p->qe - q->qb;
                        int mr = (int)(q->re - q->rb < p->re - p->rb
                                        ? q->re - q->rb : p->re - p->rb);
                        int mq = (q->qe - q->qb < p->qe - p->qb)
                                        ? q->qe - q->qb : p->qe - p->qb;
                        if (or_ > (int64_t)(mask_level_redun * mr)
                            && oq > (int)(mask_level_redun * mq)) {
                            if (p->score < q->score) {
                                p->qe = p->qb;
                                break;
                            } else {
                                q->qe = q->qb;
                            }
                        } else if (q->rb < p->rb) {
                            int w_out;
                            int score = mem_patch_reg_gpu(d_opt, d_bns, d_pac, query,
                                                          q, p, &w_out, d_buffer_ptr);
                            if (score > 0) {
                                p->n_comp += q->n_comp + 1;
                                if (q->seedcov > p->seedcov) p->seedcov = q->seedcov;
                                if (q->sub     > p->sub)     p->sub     = q->sub;
                                if (q->csub    > p->csub)    p->csub    = q->csub;
                                p->qb = q->qb;
                                p->rb = q->rb;
                                p->truesc = score;
                                p->score  = score;
                                p->w      = w_out;
                                q->qb = q->qe;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    /* --- Step 4: compact out dead entries (qe <= qb).                    */
    /* parallel for n >= SORT_PARALLEL_MIN (new_a_ptr from                 */
    /* Step 1 reused as scratch); serial thread-0 fallback for small n.    */
    __shared__ int sh_n_after;
    if (n >= SORT_PARALLEL_MIN) {
        /* new_a_ptr was allocated by Step 1; reuse as compaction scratch. */
        mem_alnreg_t *new_a = (mem_alnreg_t *)psm->new_a_ptr;
        int n_new = block_compact(a, n, new_a, compact_scan_tmp);
        /* block_compact ends with __syncthreads(); sh_n_after write is safe. */
        if (tid == 0) sh_n_after = n_new;
        __syncthreads();
    } else {
        if (tid == 0) {
            int m = 0;
            for (int i = 0; i < n; i++) {
                if (a[i].qe > a[i].qb) {
                    if (m != i) a[m] = a[i];
                    m++;
                }
            }
            sh_n_after = m;
        }
        __syncthreads();
    }
    n = sh_n_after;

    if (n > 1) {
        /* --- Step 5: sort by score DESC, rb ASC, qb ASC. */
        if (n >= SORT_PARALLEL_MIN) {
            if (tid == 0) {
                psm->new_a_ptr = CUDAKernelMalloc(
                        d_buffer_ptr, (size_t)n * sizeof(mem_alnreg_t), 8);
            }
            __syncthreads();
            mem_alnreg_t *new_a = (mem_alnreg_t *)psm->new_a_ptr;
            block_sort_by_score(a, n, sm, new_a);
        } else if (tid == 0) {
            mem_alnreg_t tmp;
            for (int i = 1; i < n; i++) {
                int     i_score = a[i].score;
                int64_t i_rb    = a[i].rb;
                int     i_qb    = a[i].qb;
                int j = i - 1;
                if (a[j].score < i_score
                    || (a[j].score == i_score
                        && (a[j].rb > i_rb
                            || (a[j].rb == i_rb && a[j].qb > i_qb)))) {
                    tmp = a[i];
                    do {
                        a[j + 1] = a[j];
                        j--;
                    } while (j >= 0
                        && (a[j].score < i_score
                            || (a[j].score == i_score
                                && (a[j].rb > i_rb
                                    || (a[j].rb == i_rb && a[j].qb > i_qb)))));
                    a[j + 1] = tmp;
                }
            }
        }
        __syncthreads();

        /* --- Step 6: parallel mark exact duplicates --- */
        for (int i = tid; i < n; i += SORT_REGIONS_BLOCK) {
            if (i >= 1) {
                if (a[i].score == a[i - 1].score
                    && a[i].rb    == a[i - 1].rb
                    && a[i].qb    == a[i - 1].qb)
                    a[i].qe = a[i].qb;
            }
        }
        __syncthreads();

        /* compact (parallel for n >= SORT_PARALLEL_MIN,                       */
        /*          serial thread-0 fallback for small n).                       */
        /* new_a_ptr was allocated by Step 5 when n >= SORT_PARALLEL_MIN;       */
        /* reuse as compaction scratch.                                          */
        if (n >= SORT_PARALLEL_MIN) {
            mem_alnreg_t *new_a = (mem_alnreg_t *)psm->new_a_ptr;
            int n_new = block_compact(a, n, new_a, compact_scan_tmp);
            if (tid == 0) sh_n_after = n_new;
            __syncthreads();
        } else {
            if (tid == 0) {
                int m = 0;
                for (int i = 0; i < n; i++) {
                    if (a[i].qe > a[i].qb) {
                        if (m != i) a[m] = a[i];
                        m++;
                    }
                }
                sh_n_after = m;
            }
            __syncthreads();
        }
        n = sh_n_after;
    }

    if (tid == 0)
        d_regs[seqID].n = n;
}


/* ------------------------------------------------------------------ */
/*  cal_max_gap helper (same as in extend.cu)                         */
/* ------------------------------------------------------------------ */
__device__ static inline int cal_max_gap_reg(const mem_opt_t *opt, int qlen)
{
    int l_del = (int)((double)(qlen * opt->a - opt->o_del) / opt->e_del + 1.);
    int l_ins = (int)((double)(qlen * opt->a - opt->o_ins) / opt->e_ins + 1.);
    int l = l_del > l_ins ? l_del : l_ins;
    l = l > 1 ? l : 1;
    return l < opt->w << 1 ? l : opt->w << 1;
}

/* ------------------------------------------------------------------ */
/*  filter_contained_regions kernel (block-cooperative parallel)       */
/*                                                                    */
/*  GPU port of bwa-mem2's post-extension seed containment check.     */
/*  After SW extension, seeds whose regions are fully contained in     */
/*  an existing higher-scoring region are purged (qe = qb).            */
/*                                                                    */
/*  Match: bwa-mem2 src/bwamem.cpp lines 3096-3197                    */
/*                                                                    */
/*  Parallelization:                                                   */
/*    - One block per read; SORT_REGIONS_BLOCK = 256 threads.          */
/*    - Step 1: CUB BlockRadixSort by score DESC (idx ASC tiebreak).   */
/*    - Step 2: outer ki block-serial; inner kj parallel OR-reduce     */
/*      (warp-OR via __shfl_xor_sync + reduce_SM[8] scratch).          */
/*    - Step 3: serial compaction (thread 0).                          */
/*                                                                    */
/*  The `lim` / `v >= lim` early-exit is dropped: the inner loop is    */
/*  an OR over kj in [0, ki-1] of an alive+containment predicate, and  */
/*  the early-exit is purely an optimization that becomes unnecessary  */
/*  once each kj is one thread's per-iteration work.                   */
/*                                                                    */
/*  Hard cap n_use = min(n, 256) preserved verbatim from the serial    */
/*  code; regions in positions [256, n) are not touched by the check   */
/*  and survive compaction unchanged.                                  */
/*                                                                    */
/*  Shared memory layout: FcrSM (FcrBlockSort temp + new_i[256] +      */
/*  reduce_SM[]).  Uses 1 item/thread (n_use ≤ 256 so                  */
/*  12-item/thread BlockRegionSort wastes ~22 registers; switch to 1/t  */
/*  drops regs from 80 to ~58 → 4 blocks/SM at 50% occupancy).         */
/*  SM: ~4 KB.                                                         */
/* ------------------------------------------------------------------ */
__launch_bounds__(256, 4)
__global__ void filter_contained_regions(
        const mem_opt_t *d_opt,
        mem_chain_v *d_chains,
        mem_alnreg_v *d_regs,
        seed_record_t *d_seed_records,
        int *d_Nseeds,
        int *d_seq_offset,
        int batch_size)
{
    const int seqID = blockIdx.x;
    if (seqID >= batch_size) return;

    int n = d_regs[seqID].n;
    if (n <= 1) return;

    const int tid = threadIdx.x;
    mem_alnreg_t *a = d_regs[seqID].a;
    const int l_query = d_seq_offset[seqID + 1] - d_seq_offset[seqID];

    /* Cap matches the serial code (idx[256] register array). */
    const int n_use = n < 256 ? n : 256;

    extern __shared__ char _fcr_SM[];
    FcrSM *sm = (FcrSM *)_fcr_SM;

    /* --- Step 1: parallel sort idx[] by score DESC, idx ASC ---
     *
     * 1 item per thread (tid ∈ [0, 255]) — n_use ≤ 256 so this covers
     * all live items.
     *
     * CUB ascending sort on packed uint64_t key:
     *   key = ((0xFFFFu - clamped_score) << 32) | idx
     * Scores at this stage are bounded by read_length * match_score
     * (worst case ~151 for E. coli, ~250 for hg38 paired-end) so clamping
     * to [0, 0xFFFF] is non-lossy.  Idx (< 256) fits in 16 bits.
     * Ascending key sort = score DESC, idx ASC tiebreak.                */
    {
        uint64_t k_arr[FCR_SORT_ITEMS];   /* 1 element */
        int      v_arr[FCR_SORT_ITEMS];
        if (tid < n_use) {
            int sc = a[tid].score;
            unsigned s = (sc < 0) ? 0u
                       : ((sc > 0xFFFF) ? 0xFFFFu : (unsigned)sc);
            k_arr[0] = ((uint64_t)(0xFFFFu - s) << 32) | (uint32_t)tid;
            v_arr[0] = tid;
        } else {
            k_arr[0] = UINT64_MAX;
            v_arr[0] = tid;
        }
        __syncthreads();
        FcrBlockSort(sm->sort_tmp).Sort(k_arr, v_arr);
        __syncthreads();
        if (tid < n_use)
            sm->new_i[tid] = (uint16_t)v_arr[0];
        __syncthreads();
    }
    /* sm->new_i[ki] = original-array idx of the ki-th score-descending region. */

    /* --- Step 2: outer ki serial, inner kj parallel OR-reduce ---
     *
     * Phase A reads a[] only (per-iter snapshot of ar fields in registers).
     * The only mutation per outer iter is thread 0 setting ar->qe = ar->qb
     * (purge), applied AFTER the reduction.  __syncthreads() at iter end
     * makes purges visible to iter ki+1.                                  */
    const int lane    = tid & 31;
    const int warpid  = tid >> 5;
    const int n_warps = SORT_REGIONS_BLOCK >> 5;   /* 8 */

    for (int ki = 0; ki < n_use; ki++) {
        const int ri = (int)sm->new_i[ki];
        mem_alnreg_t *ar = &a[ri];

        const int ar_qb = ar->qb;
        const int ar_qe = ar->qe;
        if (ar_qb >= ar_qe) {
            /* Already dead — nothing to test or write. */
            __syncthreads();
            continue;
        }
        const int64_t s_rbeg = (int64_t)ar->hash;
        const int     s_qbeg = ar->alt_sc;
        const int     s_len  = ar->seedlen0;
        const int     thr_seedlen = (int)(0.1f * (float)l_query);

        /* Phase A: each thread checks its strided kj range. */
        int local_or = 0;
        for (int kj = tid; kj < ki; kj += SORT_REGIONS_BLOCK) {
            int rj = (int)sm->new_i[kj];
            mem_alnreg_t *p = &a[rj];
            int p_qb = p->qb;
            int p_qe = p->qe;
            if (p_qb >= p_qe) continue;     /* purged earlier */

            int64_t p_rb = p->rb;
            int64_t p_re = p->re;

            /* Box containment */
            if (s_rbeg < p_rb || s_rbeg + s_len > p_re ||
                s_qbeg < p_qb || s_qbeg + s_len > p_qe) continue;

            /* Seed length tolerance vs primary seedlen0 */
            int p_seedlen0 = p->seedlen0;
            if (s_len - p_seedlen0 > thr_seedlen) continue;

            int p_w = p->w;

            /* Front distance check */
            int qd = s_qbeg - p_qb;
            int rd = (int)(s_rbeg - p_rb);
            int min_d = qd < rd ? qd : rd;
            int max_gap = cal_max_gap_reg(d_opt, min_d);
            int bw = max_gap < p_w ? max_gap : p_w;
            bool front_ok = (qd - rd < bw && rd - qd < bw);
            bool back_ok  = false;
            if (!front_ok) {
                /* Back distance check */
                qd = p_qe - (s_qbeg + s_len);
                rd = (int)(p_re - (int64_t)(s_rbeg + s_len));
                min_d = qd < rd ? qd : rd;
                max_gap = cal_max_gap_reg(d_opt, min_d);
                bw = max_gap < p_w ? max_gap : p_w;
                back_ok = (qd - rd < bw && rd - qd < bw);
            }
            if (front_ok || back_ok) { local_or = 1; }
        }

        /* Warp-OR */
        for (int m = 16; m > 0; m >>= 1)
            local_or |= __shfl_xor_sync(0xffffffff, local_or, m);
        if (lane == 0) sm->reduce_SM[warpid] = local_or;
        __syncthreads();
        if (warpid == 0) {
            int wv = (lane < n_warps) ? sm->reduce_SM[lane] : 0;
            for (int m = 4; m > 0; m >>= 1)
                wv |= __shfl_xor_sync(0xffffffff, wv, m);
            if (lane == 0) sm->reduce_SM[0] = wv;
        }
        __syncthreads();
        const int contained = sm->reduce_SM[0];

        if (contained && tid == 0) {
            ar->qe = ar->qb;    /* purge */
        }
        __syncthreads();
    }

    /* --- Step 3: compact dead entries (serial, thread 0) --- */
    __shared__ int sh_n_after;
    if (tid == 0) {
        int m = 0;
        for (int i = 0; i < n; i++) {
            if (a[i].qe > a[i].qb) {
                if (m != i) a[m] = a[i];
                m++;
            }
        }
        sh_n_after = m;
    }
    __syncthreads();
    if (tid == 0) d_regs[seqID].n = sh_n_after;
}


/* ------------------------------------------------------------------ */
/*  filter_and_score_threshold -- fused kernel                        */
/*                                                                      */
/*  Combines filter_regions (is_alt flag setting) and apply_score_filter*/
/*  (score<T compaction) into a single pass over d_regs[blockIdx.x].   */
/*  Block size: 256.  Dynamic SM: 0.                                    */
/*                                                                      */
/*  Pass 1 (all 256 threads, strided):  set is_alt for each alignment.  */
/*  Pass 2 (thread 0 only, after __syncthreads()):  compact out regions */
/*          with score < opt->T.                                         */
/*                                                                      */
/*  Must run BEFORE sort_regions so that is_alt flags are available for  */
/*  the two-pass sort.  Score compaction before sort_regions means low-  */
/*  score regions do not contribute to sub/MAPQ — this is intentional.  */
/* ------------------------------------------------------------------ */
__global__ void filter_and_score_threshold(
        const mem_opt_t *d_opt,
        const bntseq_t  *d_bns,
        mem_alnreg_v    *d_regs,
        int64_t          batch_offset)
{
    const int seqID = blockIdx.x;
    int n = d_regs[seqID].n;
    mem_alnreg_t *a = d_regs[seqID].a;
    const int tid = threadIdx.x;

    /* Pass 1: set is_alt flags (all 256 threads, stride-256 loop). */
    for (int i = tid; i < n; i += blockDim.x) {
        mem_alnreg_t *p = &a[i];
        if (p->rid >= 0 && d_bns->anns[p->rid].is_alt)
            p->is_alt = 1;
    }
    __syncthreads();

    /* Pass 2: compact out score<T regions (thread 0 serial). */
    if (tid == 0) {
        int m = 0;
        for (int i = 0; i < n; i++) {
            if (a[i].score >= d_opt->T) {
                if (i != m) a[m] = a[i];
                m++;
            }
        }
        d_regs[seqID].n = m;
    }
}

/*
 * sort_regions -- GPU port of bwa-mem2's mem_mark_primary_se.
 *
 * Launch: one block per read, SORT_REGIONS_BLOCK=256 threads/block.
 * Dynamic shared memory: sizeof(SortRegionsSM).
 *
 * Parallelism:
 *   - isort_hlt / isort_hlt2 replaced by CUB BlockRadixSort (O(n log n) parallel).
 *   - mark_primary_core replaced by block_mark_primary_core (outer-serial,
 *     inner-parallel block-cooperative, block-min reduction).
 *   - Field resets and remap loops also parallelized across 256 threads.
 *
 * Algorithm notes:
 *   - Reset stale sub/alt_sc/secondary/secondary_all; assign
 *     hash = hash_64(global_read_id+i) matching bwa-mem2.
 *   - Two-pass alt/non-alt logic with re-sort (alnreg_hlt2), remap,
 *     and second mark_primary_core pass on n_pri non-alt regions.
 *   - Sort comparator matches bwa-mem2 alnreg_hlt (score DESC, is_alt ASC,
 *     hash ASC) via packed uint64_t key for CUB ascending sort.
 */
__launch_bounds__(256, 3)
__global__ void sort_regions(
        const mem_opt_t *d_opt,
        mem_alnreg_v *d_regs,
        int n_seqs,
        int64_t batch_offset,
        void *d_buffer_pools
        )
{
    int seqID = blockIdx.x;
    if (seqID >= n_seqs) return;
    int n = d_regs[seqID].n;
    mem_alnreg_t *a = d_regs[seqID].a;
    const int tid = threadIdx.x;

    extern __shared__ char _sort_regions_SM[];
    SortRegionsSM *sm = (SortRegionsSM *)_sort_regions_SM;

    if (n <= 0) return;

    void *d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, 32 + seqID % 32);

    /* ---- Reset stale fields, assign stable hash (parallel) ---- */
    int64_t global_id = batch_offset + (int64_t)seqID;
    __shared__ int sh_n_pri;
    if (tid == 0) sh_n_pri = 0;
    __syncthreads();

    int n_iter = (n + SORT_REGIONS_BLOCK - 1) / SORT_REGIONS_BLOCK;
    int local_n_pri = 0;
    for (int k = 0; k < n_iter; k++) {
        int i = k * SORT_REGIONS_BLOCK + tid;
        if (i < n) {
            a[i].sub           = 0;
            a[i].alt_sc        = 0;
            a[i].secondary     = -1;
            a[i].secondary_all = -1;
            a[i].hash          = d_hash_64((uint64_t)(global_id + (int64_t)i));
            if (!a[i].is_alt) local_n_pri++;
        }
    }
    /* Reduce n_pri across threads */
    for (int m = 16; m > 0; m >>= 1)
        local_n_pri += __shfl_xor_sync(0xffffffff, local_n_pri, m);
    if ((tid & 31) == 0) atomicAdd(&sh_n_pri, local_n_pri);
    __syncthreads();
    int n_pri = sh_n_pri;

    if (n == 1) return;

    /* Allocate a shared temp-array pointer (used by block_sort_hlt/hlt2) */
    __shared__ mem_alnreg_t *new_a_ptr;
    __shared__ int *z_ptr;
    if (tid == 0) {
        new_a_ptr = (mem_alnreg_t *)CUDAKernelMalloc(
                d_buffer_ptr, (size_t)n * sizeof(mem_alnreg_t), 8);
    }
    __syncthreads();

    /* ---- Sort pass 1: score DESC -> is_alt ASC -> hash ASC ---- */
    block_sort_hlt(a, n, sm, new_a_ptr);

    /* ---- Pass 1: block_mark_primary_core on all n regions ---- */
    block_mark_primary_core(d_opt, n, a, sm->reduce_SM);

    /* ---- Record pass-1 ranks; capture alt_sc (parallel) ---- */
    for (int k = 0; k < n_iter; k++) {
        int i = k * SORT_REGIONS_BLOCK + tid;
        if (i < n) {
            a[i].secondary_all = i;
            if (!a[i].is_alt && a[i].secondary >= 0 && a[a[i].secondary].is_alt)
                a[i].alt_sc = a[a[i].secondary].score;
        }
    }
    __syncthreads();

    /* ---- Two-pass alt/non-alt logic ---- */
    if (n_pri > 0 && n_pri < n) {
        /* Re-sort with alnreg_hlt2 (is_alt ASC, score DESC, hash ASC).
         * sm->new_i[new_rank] = old_rank (= secondary_all rank before re-sort).
         * Build z[old_rank] = new_rank to remap secondary indices. */
        block_sort_hlt2(a, n, sm, new_a_ptr);

        if (tid == 0)
            z_ptr = (int *)CUDAKernelMalloc(d_buffer_ptr,
                                             (size_t)n * sizeof(int), 4);
        __syncthreads();
        int *z = z_ptr;

        /* Build inverse permutation: z[old_rank] = new_rank */
        for (int k = 0; k < n_iter; k++) {
            int new_rank = k * SORT_REGIONS_BLOCK + tid;
            if (new_rank < n)
                z[(int)sm->new_i[new_rank]] = new_rank;
        }
        __syncthreads();

        /* Remap secondary indices (parallel) */
        for (int k = 0; k < n_iter; k++) {
            int i = k * SORT_REGIONS_BLOCK + tid;
            if (i < n) {
                if (a[i].secondary >= 0) {
                    a[i].secondary_all = z[a[i].secondary];
                    if (a[i].is_alt) a[i].secondary = 0x7fffffff; /* INT_MAX for alt */
                } else {
                    a[i].secondary_all = -1;
                }
            }
        }
        __syncthreads();

        /* Reset sub/secondary for non-alt (first n_pri) regions before pass-2 */
        int n_iter2 = (n_pri + SORT_REGIONS_BLOCK - 1) / SORT_REGIONS_BLOCK;
        for (int k = 0; k < n_iter2; k++) {
            int i = k * SORT_REGIONS_BLOCK + tid;
            if (i < n_pri) {
                a[i].sub       = 0;
                a[i].secondary = -1;
            }
        }
        __syncthreads();

        /* Re-run primary marking on only n_pri non-alt regions */
        block_mark_primary_core(d_opt, n_pri, a, sm->reduce_SM);
    } else {
        /* No mixed alt/non-alt: secondary_all = secondary from pass 1 */
        for (int k = 0; k < n_iter; k++) {
            int i = k * SORT_REGIONS_BLOCK + tid;
            if (i < n)
                a[i].secondary_all = a[i].secondary;
        }
    }
}
