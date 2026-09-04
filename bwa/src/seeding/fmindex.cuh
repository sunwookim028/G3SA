#ifndef _FMINDEX_CUH
#define _FMINDEX_CUH

#include "bwa.h"  // defines CP_OCC, CP_SHIFT, CP_MASK (bwa.h lines 79+)

// SA lookup: sets *rb as suffixArray[k] (works for compressed SA)
extern __device__ void sa_lookup(const fmindex_t *devFmIndex, uint64_t k, int64_t *rb);

// KMER hash key: encode 12-mer at s[0..KMER_K-1] (2-bit uint8_t); returns -1 if any base is N (>=4)
extern __device__ int devicehashK(const uint8_t *s);

// LF mapping
extern __device__ void LFMap(const fmindex_t *devFmIndex, uint64_t k, uint8_t bwt_b, uint64_t *next_k);

// Single-base backward extension (4-way)
extern __device__ void backwardExt(const int64_t sentinel_index, const bwtintv_t *smem, uint8_t base, bwtintv_t *nextSmem, const uint64_t *d_one_hot, const CP_OCC *d_cp_occ, const int64_t *d_count);

// Dual-base backward extension (16-way)
extern __device__ void backwardExt2(const int64_t sentinel_index, const bwtintv_t *smem, uint8_t base0, uint8_t base1, bwtintv_t *nextSmem, const uint64_t *d_one_hot, const CP_OCC *d_cp_occ, const int64_t *d_count, const CP_OCC2 *d_cp_occ2, const int64_t *d_count2, const uint8_t *d_first_base);

// Single-base backward extension (backward direction)
extern __device__ void backwardExtBackward(const int64_t sentinel_index, const bwtintv_t *smem, uint8_t base, bwtintv_t *nextSmem, const uint64_t *d_one_hot, const CP_OCC *d_cp_occ, const int64_t *d_count);

// Dual-base backward extension (backward direction)
extern __device__ void backwardExt2Backward(const int64_t sentinel_index, const bwtintv_t *smem, uint8_t base0, uint8_t base1, bwtintv_t *nextSmem, const uint64_t *d_one_hot, const CP_OCC *d_cp_occ, const int64_t *d_count, const CP_OCC2 *d_cp_occ2, const int64_t *d_count2, const uint8_t *d_first_base);

// ==========================================================================
// SM OCC Cache helpers — inline device functions for reseedV2.
// Guarded by G3_SM_OCC_CACHE (off by default; enable via -DG3_SM_OCC_CACHE).
//
// Layout: direct-mapped, G3_OCC_CACHE_SETS=64 sets, one CP_OCC per set.
//   SM cost: 64 × (8B key + 64B CP_OCC val) = 4608 B per warp.
//   16 blocks/SM × 4608 B = 73.7 KB < 100 KB A6000 SM limit per SM.
//   NOTE: RTX A6000 (sm_86) has 100 KB SM/SM max. 128 entries (9216 B × 16 = 147.5 KB)
//   would exceed this limit and reduce to 10 blocks/SM — a regression.
//   64 is the largest power-of-2 that keeps 16 blocks/SM.
//
// Usage pattern (lane 0, serial — no __syncthreads needed):
//   1. Declare in kernel: __shared__ int64_t _keys[WAYS][64]; __shared__ CP_OCC _vals[WAYS][64];
//   2. Init:  for (int i=0; i<64; i++) cache_keys[i] = G3_OCC_CACHE_INVALID;
//   3. Call:  backwardExt_c(..., cache_keys, cache_vals) / backwardExtBackward_c(...)
//   4. Use FORWARD_EXT_C / BACKWARD_EXT_C macros (defined in macro.h).
// ==========================================================================
#ifdef G3_SM_OCC_CACHE

#define G3_OCC_CACHE_SETS    64
#define G3_OCC_CACHE_INVALID (-1LL)

static __device__ __forceinline__ CP_OCC g3_get_occ_cached(
        const CP_OCC * __restrict__ global_occ,
        int64_t occ_id,
        int64_t * __restrict__ cache_keys,
        CP_OCC  * __restrict__ cache_vals)
{
    int set = (int)(occ_id & (G3_OCC_CACHE_SETS - 1));
    if (cache_keys[set] == occ_id)
        return cache_vals[set];
    CP_OCC v = global_occ[occ_id];
    cache_keys[set] = occ_id;
    cache_vals[set] = v;
    return v;
}

// Cached backwardExt: 4-way extension with SM OCC cache.
static __device__ __forceinline__ void backwardExt_c(
        const int64_t sentinel_index,
        const bwtintv_t *smem, uint8_t base, bwtintv_t *nextSmem,
        const uint64_t *d_one_hot, const CP_OCC *d_cp_occ, const int64_t *d_count,
        int64_t *cache_keys, CP_OCC *cache_vals)
{
    uint8_t b;
    int64_t k[4], l[4], s[4];
    for (b = 0; b < 4; b++)
    {
        int64_t sp = (int64_t)(smem->x[0]) - 1;
        int64_t ep = (int64_t)(smem->x[0]) + (int64_t)(smem->x[2]) - 1;
        CP_OCC  entry_sp = g3_get_occ_cached(d_cp_occ, sp >> CP_SHIFT, cache_keys, cache_vals);
        int64_t occ_sp   = entry_sp.cp_count[b];
        occ_sp += __popcll(entry_sp.one_hot_bwt_str[b] & d_one_hot[sp & CP_MASK]);
        CP_OCC  entry_ep = g3_get_occ_cached(d_cp_occ, ep >> CP_SHIFT, cache_keys, cache_vals);
        int64_t occ_ep   = entry_ep.cp_count[b];
        occ_ep += __popcll(entry_ep.one_hot_bwt_str[b] & d_one_hot[ep & CP_MASK]);
        k[b] = d_count[b] + occ_sp;
        s[b] = occ_ep - occ_sp;
    }
    int64_t sentinel_offset = 0;
    if ((smem->x[0] <= sentinel_index) && ((smem->x[0] + smem->x[2]) > sentinel_index))
        sentinel_offset = 1;
    l[3] = smem->x[1] + sentinel_offset;
    l[2] = l[3] + s[3];
    l[1] = l[2] + s[2];
    l[0] = l[1] + s[1];
    nextSmem->x[0] = k[base];
    nextSmem->x[1] = l[base];
    nextSmem->x[2] = s[base];
}

// Cached backwardExtBackward: single-base extension (backward direction).
static __device__ __forceinline__ void backwardExtBackward_c(
        const int64_t sentinel_index,
        const bwtintv_t *smem, uint8_t base, bwtintv_t *nextSmem,
        const uint64_t *d_one_hot, const CP_OCC *d_cp_occ, const int64_t *d_count,
        int64_t *cache_keys, CP_OCC *cache_vals)
{
    int64_t sp = (int64_t)(smem->x[0]) - 1;
    int64_t ep = (int64_t)(smem->x[0]) + (int64_t)(smem->x[2]) - 1;
    CP_OCC  entry_sp = g3_get_occ_cached(d_cp_occ, sp >> CP_SHIFT, cache_keys, cache_vals);
    int64_t occ_sp   = entry_sp.cp_count[base];
    occ_sp += __popcll(entry_sp.one_hot_bwt_str[base] & d_one_hot[sp & CP_MASK]);
    CP_OCC  entry_ep = g3_get_occ_cached(d_cp_occ, ep >> CP_SHIFT, cache_keys, cache_vals);
    int64_t occ_ep   = entry_ep.cp_count[base];
    occ_ep += __popcll(entry_ep.one_hot_bwt_str[base] & d_one_hot[ep & CP_MASK]);
    nextSmem->x[0] = (uint64_t)(d_count[base] + occ_sp);
    nextSmem->x[2] = (uint64_t)(occ_ep - occ_sp);
}

#endif /* G3_SM_OCC_CACHE */

#endif /* _FMINDEX_CUH */
