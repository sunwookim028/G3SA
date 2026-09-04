#include "bwa.h"
#include "macro.h"
#include "gmem_alloc.cuh"
#include "fmindex.cuh"
#include "seed.cuh"
#include <string.h>
#include <cub/cub.cuh>
#include <chrono>
#include <stdio.h>

//
// Preseed
//

// forward declaration
__device__ void preseedOnePos2Backward(const fmindex_t *fmi, int read_len, const uint8_t *read, int m, int min_seed_len, bwtintv_t *preSeeds, kmers_bucket_t *d_kmersHashTab);

__global__ void preseed_and_filter(
        const fmindex_t  *devFmIndex,
        const mem_opt_t *d_opt,
        const uint8_t *d_seq,
        int *d_seq_offset,
        smem_aux_t *d_aux, 			// aux output
        kmers_bucket_t *d_kmerHashTab,
        void *d_buffer_pools)
{
    int seq_id = blockIdx.x;
    const uint8_t *read;
    int minSeedLen;     // option: minimum seed length
    int n; // local var
    int read_len; // read sequence length

    int seq_offset = d_seq_offset[seq_id];
    int seq_offset_next = d_seq_offset[seq_id + 1];
    read = d_seq + seq_offset;
    read_len = seq_offset_next - seq_offset;
    minSeedLen = d_opt->min_seed_len;

    __shared__ bwtintv_t sharedPreSeeds[MAX_LEN_READ];
    __shared__ uint8_t sharedRead[MAX_LEN_READ];
    for (int j = threadIdx.x; j<read_len; j+=blockDim.x)
    {
        sharedRead[j] = read[j];
        sharedPreSeeds[j].info = 0;
    }
    __syncthreads(); __syncwarp();


    // Collect preSeed = read[m..n]
    // n loop is parallelized.
    for(n = minSeedLen - 1 + threadIdx.x; n < read_len; n += blockDim.x)
    {
        preseedOnePos2Backward(devFmIndex, read_len, sharedRead, n, minSeedLen, sharedPreSeeds, d_kmerHashTab);
    }
    __syncthreads(); __syncwarp();


    // Inspect in parallel 
	__shared__ bool sharedIsSeed[MAX_LEN_READ];
    for(n = threadIdx.x; n < read_len - 1; n += blockDim.x)
    {
        sharedIsSeed[n] = (bool)(sharedPreSeeds[n].info);
		if(((sharedPreSeeds[n].info >> 32) == (sharedPreSeeds[n + 1].info >> 32)) && (sharedPreSeeds[n + 1].info != 0)) // non-super seed
        {
            sharedIsSeed[n] = 0;
        }
	}
	__syncthreads(); __syncwarp();

    // Parallel gather: prefix sum + parallel scatter
    __shared__ int S_prefix[MAX_LEN_READ];
    __shared__ int S_total_seeds;
    __shared__ bwtintv_t *S_gm_seeds;

    // Handle the last element (not covered by the filter loop above)
    if (threadIdx.x == 0) {
        sharedIsSeed[read_len - 1] = (bool)(sharedPreSeeds[read_len - 1].info);
    }
    __syncthreads();

    // Thread 0 computes exclusive prefix sum (O(read_len) ~ 150, very fast)
    if (threadIdx.x == 0) {
        int count = 0;
        for (int i = 0; i < read_len; i++) {
            S_prefix[i] = count;
            if (sharedIsSeed[i]) count++;
        }
        S_total_seeds = count;
    }
    __syncthreads();

    // Thread 0 allocates global memory (CUDAKernelMalloc is not thread-safe)
    if (threadIdx.x == 0) {
        d_aux[blockIdx.x].mem.n = S_total_seeds;
        void *d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, blockIdx.x % 32);
        S_gm_seeds = (bwtintv_t*)CUDAKernelMalloc(d_buffer_ptr, sizeof(bwtintv_t) * S_total_seeds, sizeof(bwtintv_t));
        d_aux[blockIdx.x].mem.a = S_gm_seeds;
    }
    __syncthreads();

    // All threads scatter in parallel using prefix sum offsets
    for (int i = threadIdx.x; i < read_len; i += blockDim.x) {
        if (sharedIsSeed[i]) {
            S_gm_seeds[S_prefix[i]] = sharedPreSeeds[i];
        }
    }
}


// Collect a preSeed = read[m..n]
template <bool STRIDED>
__device__ void preseedOnePosBackwardImpl(const fmindex_t *fmi, int read_len, const uint8_t *read, int n, int minSeedLen, bwtintv_t *preseeds, kmers_bucket_t *d_kmersHashTab)
{
    uint64_t *oneHot = fmi->oneHot;
    int64_t *count = fmi->count;
    int64_t *count2 = fmi->count2;
    CP_OCC *cpOcc = fmi->cpOcc;
    CP_OCC2 *cpOcc2 = fmi->cpOcc2;
    int64_t sentinelIndex = *(fmi->sentinelIndex);
    uint8_t firstBase = *(fmi->firstBase);

    bwtintv_t smem, oldSmem;
    const int min_intv = 1;
    if (read[n] >= 4)
    {
        preseeds[n].info = 0;
        return;
    }
    smem.x[0] = count[read[n]];
    smem.x[1] = count[3 - read[n]];
    smem.x[2] = count[read[n] + 1] - count[read[n]];

    if (!STRIDED) {
        int m;
        uint8_t base0;

        // Elongate
        for(m = n; m >= 1; m--)
        {
            base0 = read[m - 1];
            if(base0 >= 4)
            {
                break;
            }

            oldSmem = smem;
            BACKWARD_EXT_B(smem, base0);
            if(smem.x[2] < min_intv)
            {
                smem = oldSmem;
                break;
            }
        }

        // Check
        if(n - m + 1 >= minSeedLen)
        {
            smem.info = INFO(m, n);
        }
        else
        {
            smem.info = 0;
        }

        // Collect
        preseeds[n] = smem;
        return;
    }

    int m;
    uint8_t base0, base1;

    // Extend
    for(m = n; m > 1; m -= 2) {   // at the start, smem == read[m, n]
        oldSmem = smem;
        base0 = read[m - 1];
        base1 = read[m - 2];
        if(base1 >= 4) {
            if(base0 < 4) {
                BACKWARD_EXT_B(smem, base0);
                if(smem.x[2] < min_intv) {
                    smem = oldSmem;
                } else {
                    m--;
                }
            }
            break; // smem == read[m,n]
        }

        if(base0 >= 4) {
            break; // smem == read[m,n]
        }

        BACKWARD_EXT2_B(smem, base0, base1);
        if(smem.x[2] < min_intv) {
            smem = oldSmem;
            BACKWARD_EXT_B(smem, base0);
            if(smem.x[2] < min_intv) {
                smem = oldSmem;
            } else {
                m--;
            }
            break;
        }
    }
    if(m == 1) {
        oldSmem = smem;
        if(read[0] < 4) {
            BACKWARD_EXT_B(smem, read[0]);
            if(smem.x[2] < min_intv) {
                smem = oldSmem;
            } else {
                m--;
            }
        }
    }

    if(n - m + 1 >= minSeedLen) {
        smem.info = INFO(m, n);
    } else {
        smem.info = 0;
    }

    preseeds[n] = smem;
}

__device__ void preseedOnePos2Backward(const fmindex_t *fmi, int read_len, const uint8_t *read, int n, int min_seed_len, bwtintv_t *preSeeds, kmers_bucket_t *d_kmersHashTab)
{
    preseedOnePosBackwardImpl<true>(fmi, read_len, read, n, min_seed_len, preSeeds, d_kmersHashTab);
}

//
// Re-Seed
//

__device__ void reseedThirdRound(const fmindex_t *fmi, const uint8_t *sharedRead, \
        int read_len, int minSeedLen, int maxIntervalSize,\
        bwtintv_t *seeds, int *num_seeds3);


// ==========================================================================
// reseedV2 — 1-warp-per-block (<<<batch_size, 32>>>).  Lane 0 drives the
// serial FM-index walk; the other 31 lanes early-exit.
//
// On CUDA < 13, an inline asm pragma enables smem_spilling, directing the
// compiler to spill registers to fast shared memory instead of slow
// local/global memory.
//
// ABI note: backwardExt() is 86 regs as extern __device__; entry function must
//   satisfy max_regs >= 86. __lb__(32,16) gives max_regs=128 >= 86.
// ==========================================================================
__launch_bounds__(32, 16)
__global__ void reseedV2(
        const fmindex_t *devFmIndex,
        const mem_opt_t *d_opt,
        uint8_t *d_seq,
        int *d_seq_offset,
        smem_aux_t *d_aux, 			// aux output
        kmers_bucket_t *d_kmerHashTab,
        void * d_buffer_pools,
        int num_reads
        )
{
#if __CUDACC_VER_MAJOR__ < 13
    asm volatile (".pragma \"enable_smem_spilling\";");
#endif
    const int lane   = threadIdx.x & 31;
    int seq_id = blockIdx.x;
    if(seq_id >= num_reads)
    {
        return;
    }
    if(lane != 0)
    {
        return;
    }

    int num_smem, max_num_seed3;
    int min_seed_len, min_seed_intv;
    int read_len, pivot_pos;
    int num_allseeds, cap_allseeds;
#define LM_CAP 16 // local-memory staging capacity for LEP/seed arrays; spills to global memory beyond this
    bwtintv_t lm_seeds[LM_CAP];
    bwtintv_t lm_leps[LM_CAP];
    void *d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, seq_id%32);
#ifdef G3_SM_OCC_CACHE
    const int g3_warp_id = 0;  // single warp per block
#endif
    bwtintv_t *seeds;
    bwtintv_t *leps;
    int num_seeds, num_leps;
    int cap_seeds, cap_leps;

    bwtintv_t *smem;
    int split_len, split_intv;
    int k;
    int seq_offset, seq_offset_next;
    seq_offset = d_seq_offset[seq_id];
    seq_offset_next = d_seq_offset[seq_id + 1];
    read_len = seq_offset_next - seq_offset;
    const uint8_t *read = d_seq + seq_offset;

    uint64_t *oneHot = devFmIndex->oneHot; // DO NOT change names for macros
    int64_t *count = devFmIndex->count;
    int64_t *count2 = devFmIndex->count2;
    CP_OCC *cpOcc = devFmIndex->cpOcc;
    CP_OCC2 *cpOcc2 = devFmIndex->cpOcc2;
    int64_t sentinelIndex = *(devFmIndex->sentinelIndex);
    uint8_t firstBase = *(devFmIndex->firstBase);

    // SM OCC cache: direct-mapped, 64 sets, per-warp.
    // 1 warp/block: 1 × 64 sets × (8+64)B = 4608 B shared memory.
    // 16 blocks/SM × 4608 B = 73.7 KB < 100 KB A6000 SM/SM limit.
    // Off by default; enable with -DG3_SM_OCC_CACHE.
#ifdef G3_SM_OCC_CACHE
    __shared__ int64_t _occ_cache_keys[1][G3_OCC_CACHE_SETS];
    __shared__ CP_OCC  _occ_cache_vals[1][G3_OCC_CACHE_SETS];
    int64_t *cache_keys = _occ_cache_keys[g3_warp_id];
    CP_OCC  *cache_vals = _occ_cache_vals[g3_warp_id];
#endif /* G3_SM_OCC_CACHE */

    num_smem = d_aux[seq_id].mem.n;
    split_len = (int)(d_opt->min_seed_len * d_opt->split_factor \
                        + .499); // default: 19 * 1.5 + .499 = 28
    split_intv = d_opt->split_width; // default: 10
    min_seed_len = d_opt->min_seed_len;

    {
        if(num_smem == 0) {
            return;
        }
#ifdef G3_SM_OCC_CACHE
        // Initialize cache to invalid (lane 0, serial — no __syncthreads needed).
        for (int _ci = 0; _ci < G3_OCC_CACHE_SETS; _ci++)
            cache_keys[_ci] = G3_OCC_CACHE_INVALID;
#endif /* G3_SM_OCC_CACHE */
        bwtintv_t *new_allseeds;
        num_allseeds = num_smem;
        cap_allseeds = num_smem << 1;
        new_allseeds = (bwtintv_t*)CUDAKernelMalloc(d_buffer_ptr,\
                sizeof(bwtintv_t) * cap_allseeds, sizeof(bwtintv_t));
        for(int l=0; l<num_allseeds; l++)
        {
            new_allseeds[l] = d_aux[seq_id].mem.a[l];
        }
        d_aux[seq_id].mem.a = new_allseeds;
    }

    for(k = 0; k < num_smem; k++)
    {
        smem = d_aux[seq_id].mem.a + k;
        if(PLEN(smem) < split_len || smem->x[2] > split_intv) // not long
        {
            continue;
        }
        // reseed
        min_seed_intv = smem->x[2] + 1;
        pivot_pos = (PM(smem) + PN(smem) + 1) >> 1;
        if(read[pivot_pos] >=4)
        {
            continue;
        }
        num_seeds = num_leps = 0;
        seeds = lm_seeds; // if num_{seeds, leps} exceeds LM_CAP, spill all to gm.
        leps = lm_leps;
        cap_seeds = cap_leps = LM_CAP;

        // 1. collect LEPs
        int m, n;
        bwtintv_t seed, oldSeed;

        // KMER hash init — skip KMER_K BWT forward-extension steps by looking up
        // the prebuilt KMER-K=12 hash table.
        // Safe: any LEP ending before pivot_pos+KMER_K has length < KMER_K=12 < min_seed_len=19
        // and would be dropped anyway; we can skip those steps without missing any valid LEP.
        {
            int kh_key = -1;
            if (pivot_pos + KMER_K <= read_len && d_kmerHashTab != NULL)
                kh_key = devicehashK(read + pivot_pos);
            if (kh_key >= 0) {
                kmers_bucket_t kh = d_kmerHashTab[kh_key];
                seed.x[0] = kh.x[0];
                seed.x[1] = kh.x[1];
                seed.x[2] = kh.x[2];
                seed.info = INFO(pivot_pos, pivot_pos + KMER_K - 1);
                n = pivot_pos + KMER_K - 1;
            } else {
                seed.x[0] = count[read[pivot_pos]];
                seed.x[1] = count[3 - read[pivot_pos]];
                seed.x[2] = count[read[pivot_pos] + 1] - count[read[pivot_pos]];
                seed.info = INFO(pivot_pos, pivot_pos);
                n = pivot_pos;
            }
        }

        for(; n < read_len - 1; n++) // collect lep = read[pivot_pos..n]
        {
            oldSeed = seed;
            if(read[n + 1] >= 4)
            {
                break;
            }

#ifdef G3_SM_OCC_CACHE
            FORWARD_EXT_C(seed, read[n + 1]);
#else
            FORWARD_EXT(seed, read[n + 1]);
#endif
            seed.info = INFO(pivot_pos, n + 1);

            if(seed.x[2] < oldSeed.x[2])
            {
                if(num_leps == cap_leps)
                {
                    bwtintv_t *new_leps;
                    new_leps = (bwtintv_t*)CUDAKernelMalloc(d_buffer_ptr,\
                            sizeof(bwtintv_t) * cap_leps << 1, sizeof(bwtintv_t));
                    for(int l=0; l<num_leps; l++)
                    {
                        new_leps[l] = leps[l];
                    }
                    cap_leps = cap_leps << 1;
                    leps = new_leps;
                }
                leps[num_leps++] = oldSeed;
                if(seed.x[2] < min_seed_intv)
                {
                    break;
                }
            }
        }
        if(seed.x[2] >= min_seed_intv)
        {
            seed.info = INFO(pivot_pos, n);
            if(num_leps == cap_leps)
            {
                bwtintv_t *new_leps;
                new_leps = (bwtintv_t*)CUDAKernelMalloc(d_buffer_ptr,\
                        sizeof(bwtintv_t) * cap_leps << 1, sizeof(bwtintv_t));
                for(int l=0; l<num_leps; l++)
                {
                    new_leps[l] = leps[l];
                }
                cap_leps = cap_leps << 1;
                leps = new_leps;
            }
            leps[num_leps++] = seed;
        }

        // reverse LEPs
        for(int j = 0; j < (num_leps >> 1); j++)
        {
            bwtintv_t temp = leps[j];
            leps[j] = leps[num_leps - 1 - j];
            leps[num_leps - 1 - j] = temp;
        }

        // 2.backward Ext LEPs
        // Attempt 2-base steps (BACKWARD_EXT2_B) where both bases are
        // valid (m >= 2, read[m-2] < 4, read[m-1] < 4).  backwardExt2Backward
        // uses the original (sp, ep) for both CP_OCC2 lookups — no intermediate
        // interval needed — so both DRAM requests issue in parallel, halving
        // effective latency vs two sequential 1-base calls.
        // Fall back to 1-base (BACKWARD_EXT_C) when m == 1 or either base is N.
        // All surviving LEPs always share the same m after each outer step so
        // the loop invariant (lep.x[2] >= min_seed_intv for all leps[]) holds.
        bwtintv_t lep;
        int currInterval;
        uint8_t base, base0, base1;
        // A loop invariant: at the start of each loop,
        // all leps satisfy the minimum intv size. i.e. lep.x[2] >= min_seed_intv.
        bwtintv_t oldLep;
        m = pivot_pos;
        while(m > 0) // collect seed = read[m..n]
        {
            // Determine if we can take a 2-base step.
            bool two_base = false;
            if(m >= 2) {
                base0 = read[m - 1]; // closer base (appended first)
                base1 = read[m - 2]; // further base (appended second)
                if(base0 < 4 && base1 < 4) {
                    two_base = true;
                }
            }
            if(!two_base) {
                base = read[m - 1];
                if(base >= 4) {
                    break;
                }
            }

            int num_old_leps = num_leps;
            currInterval = min_seed_intv;
            int j = 0; num_leps = 0;
            bool collected = false;
            if(two_base) {
                // 2-base step: advance m by 2.
                // Guard: backwardExt2Backward computes sp = x[0]-1 and indexes
                // d_cp_occ2[sp >> CP_SHIFT]. If x[0] == 0 then sp = -1 and the
                // load would be out-of-bounds. Treat such LEPs as having failed.
                while(j < num_old_leps) {
                    oldLep = lep = leps[j];
                    if(lep.x[0] == 0) { lep.x[2] = 0; }
                    else { BACKWARD_EXT2_B(lep, base0, base1); }
                    if(lep.x[2] >= currInterval) {
                        lep.info = INFO(m-2, N(lep));
                        leps[num_leps++] = lep;
                        currInterval = lep.x[2];
                    } else {
                        if(!collected && LEN(oldLep) >= min_seed_len) {
                            if(num_seeds == cap_seeds)
                            {
                                bwtintv_t *new_seeds = (bwtintv_t*)CUDAKernelMalloc(d_buffer_ptr,\
                                        sizeof(bwtintv_t) * cap_seeds << 1, sizeof(bwtintv_t));
                                cap_seeds = cap_seeds << 1;
                                for(int l=0; l<num_seeds; l++) {
                                    new_seeds[l] = seeds[l];
                                }
                                seeds = new_seeds;
                            }
                            seeds[num_seeds++] = oldLep;
                            collected = true;
                        }
                    }
                    j++;
                }
                m -= 2;
            } else {
                // 1-base step: advance m by 1.
                while(j < num_old_leps) {
                    oldLep = lep = leps[j];
#ifdef G3_SM_OCC_CACHE
                    BACKWARD_EXT_C(lep, base);
#else
                    BACKWARD_EXT(lep, base);
#endif
                    if(lep.x[2] >= currInterval) {
                        lep.info = INFO(m-1, N(lep));
                        leps[num_leps++] = lep;
                        currInterval = lep.x[2];
                    } else {
                        if(!collected && LEN(oldLep) >= min_seed_len) {
                            if(num_seeds == cap_seeds)
                            {
                                bwtintv_t *new_seeds = (bwtintv_t*)CUDAKernelMalloc(d_buffer_ptr,\
                                        sizeof(bwtintv_t) * cap_seeds << 1, sizeof(bwtintv_t));
                                cap_seeds = cap_seeds << 1;
                                for(int l=0; l<num_seeds; l++) {
                                    new_seeds[l] = seeds[l];
                                }
                                seeds = new_seeds;
                            }
                            seeds[num_seeds++] = oldLep;
                            collected = true;
                        }
                    }
                    j++;
                }
                m -= 1;
            }
            if(num_leps == 0) {
                break;
            }
        }
        if(num_leps > 0) {
            oldLep = leps[0];
            if(LEN(oldLep) >= min_seed_len) {
                seeds[num_seeds++] = oldLep;
            }
        }
        // collected all seed2s from this seed


        if(cap_allseeds - num_allseeds < num_seeds)
        {
            bwtintv_t *new_allseeds;
            new_allseeds = (bwtintv_t*)CUDAKernelMalloc(d_buffer_ptr,\
                    sizeof(bwtintv_t) * (num_allseeds + num_seeds) << 1, sizeof(bwtintv_t));
            for(int l=0; l<num_allseeds; l++)
            {
                new_allseeds[l] = d_aux[seq_id].mem.a[l];
            }
            cap_allseeds = (num_allseeds + num_seeds) << 1;
            d_aux[seq_id].mem.a = new_allseeds;
        }

        for(int l=0; l<num_seeds; l++)
        {
            d_aux[seq_id].mem.a[num_allseeds++] = seeds[l];
        }
    }

    // reallocate memory to store reseeded seeds
    max_num_seed3 = read_len / (d_opt->min_seed_len + 1) + 1;
    if(num_allseeds + max_num_seed3 + 1 > cap_allseeds)
    {
        bwtintv_t *new_allseeds;
        new_allseeds = (bwtintv_t*)CUDAKernelMalloc(d_buffer_ptr,\
                sizeof(bwtintv_t) * (num_allseeds + max_num_seed3 + 1), sizeof(bwtintv_t));
        for(int l=0; l<num_allseeds; l++)
        {
            new_allseeds[l] = d_aux[seq_id].mem.a[l];
        }
        d_aux[seq_id].mem.a = new_allseeds;
    }
    d_aux[seq_id].mem.n = num_allseeds;
}

__device__ void reseedThirdRound(const fmindex_t *fmi, const uint8_t *read, \
        int read_len, int minSeedLen, int maxIntervalSize,\
        bwtintv_t *seeds, int *num_seeds3)
{
    uint64_t *oneHot = fmi->oneHot;
    int64_t *count = fmi->count;
    CP_OCC *cpOcc = fmi->cpOcc;
    int64_t sentinelIndex = *(fmi->sentinelIndex);

    int m = 0;
    int n = 0;
    int num_seeds = 0;

    while(m < read_len)
    {
        int next_m = m + 1;

        bwtintv_t seed;
        uint8_t base = (uint8_t)read[m];

        if(base < 4)
        {
            seed.x[0] = count[base];
            seed.x[1] = count[3 - base];
            seed.x[2] = count[base + 1] - count[base];

            for(n = m + 1; n < read_len; n++)
            {
                next_m = n + 1;
                base = (uint8_t)read[n];
                if(base < 4)
                {
                    FORWARD_EXT(seed, base);
                    if((seed.x[2] < maxIntervalSize) && (n-m+1) >= minSeedLen)
                    {
                        if(seed.x[2] > 0)
                        {
                            seed.info = INFO(m,n);
                            seeds[num_seeds++] = seed;
                        }
                        break;
                    }
                }
                else
                {
                    break;
                }
            }
        }
        m = next_m;
    }
    *num_seeds3 = num_seeds;
}

//
// SA -> Rbeg
//

/* Cached variant of reseedThirdRound — same linear FM scan but uses SM OCC cache.
 * Requires cache_keys and cache_vals parameters (per-block SM arrays); the
 * FORWARD_EXT_C macro references those names directly.
 * Only compiled when G3_SM_OCC_CACHE is defined. */
#ifdef G3_SM_OCC_CACHE
__device__ void reseedThirdRoundCached(
        const fmindex_t *fmi, const uint8_t *read,
        int read_len, int minSeedLen, int maxIntervalSize,
        bwtintv_t *seeds, int *num_seeds3,
        int64_t *cache_keys, CP_OCC *cache_vals)
{
    uint64_t *oneHot         = fmi->oneHot;
    int64_t  *count          = fmi->count;
    CP_OCC   *cpOcc          = fmi->cpOcc;
    int64_t   sentinelIndex  = *(fmi->sentinelIndex);

    int m = 0, n = 0, num_seeds = 0;
    while (m < read_len) {
        int next_m = m + 1;
        bwtintv_t seed;
        uint8_t base = (uint8_t)read[m];
        if (base < 4) {
            seed.x[0] = count[base];
            seed.x[1] = count[3 - base];
            seed.x[2] = count[base + 1] - count[base];
            for (n = m + 1; n < read_len; n++) {
                next_m = n + 1;
                base = (uint8_t)read[n];
                if (base < 4) {
                    FORWARD_EXT_C(seed, base);
                    if ((seed.x[2] < maxIntervalSize) && (n - m + 1) >= minSeedLen) {
                        if (seed.x[2] > 0) {
                            seed.info = INFO(m, n);
                            seeds[num_seeds++] = seed;
                        }
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        m = next_m;
    }
    *num_seeds3 = num_seeds;
}
#endif /* G3_SM_OCC_CACHE */

__launch_bounds__(32, 16)
__global__ void reseedLastRound(
        const fmindex_t *devFmIndex,
        const mem_opt_t *d_opt,
        uint8_t *d_seq,
        int *d_seq_offset,
        smem_aux_t *d_aux,
        kmers_bucket_t *d_kmerHashTab,
        int numReads)
{
#if __CUDACC_VER_MAJOR__ < 13
    asm volatile (".pragma \"enable_smem_spilling\";");
#endif
    // Warp-per-read launch: <<<batch_size, 32>>>, one warp == one read.
    // Lane 0 drives the short serial call; other lanes early-exit. Body
    // is just a helper invocation, no cooperative opportunity.
    int seq_id = blockIdx.x;
    if(seq_id >= numReads)
    {
        return;
    }
    if((threadIdx.x & 31) != 0)
    {
        return;
    }

    int num_seeds, num_seeds3;
    num_seeds = d_aux[seq_id].mem.n;
    bwtintv_t *seeds3 = d_aux[seq_id].mem.a + num_seeds;

    // Guard: if no seed buffer was allocated (e.g. all-N read), skip
    if (d_aux[seq_id].mem.a == NULL) return;

    const uint8_t *read;
    int read_len;

    int seq_offset = d_seq_offset[seq_id];
    int seq_offset_next = d_seq_offset[seq_id + 1];
    read = d_seq + seq_offset;
    read_len = seq_offset_next - seq_offset;

    int minSeedLen = d_opt->min_seed_len + 1;
    int maxIntervalSize = d_opt->max_mem_intv;

    // SM OCC cache for reseedThirdRound: same pattern as reseedV2.
    // 64-entry direct-mapped cache, per-block (lane 0 only in this kernel).
    // SM cost: 64×8B (keys) + 64×64B (vals) = 4608 B.
    // 16 blocks/SM × 4608 B = 73.7 KB < 100 KB A6000 SM/SM limit.
#ifdef G3_SM_OCC_CACHE
    __shared__ int64_t _r3_cache_keys[G3_OCC_CACHE_SETS];
    __shared__ CP_OCC  _r3_cache_vals[G3_OCC_CACHE_SETS];
    for (int _ci = 0; _ci < G3_OCC_CACHE_SETS; _ci++)
        _r3_cache_keys[_ci] = G3_OCC_CACHE_INVALID;
    reseedThirdRoundCached(devFmIndex, read, read_len, minSeedLen, maxIntervalSize,
                           seeds3, &num_seeds3, _r3_cache_keys, _r3_cache_vals);
#else
    reseedThirdRound(devFmIndex, read, read_len, minSeedLen, maxIntervalSize,
                     seeds3, &num_seeds3);
#endif
    d_aux[seq_id].mem.n = num_seeds + num_seeds3;
}


// input: mem intervals
// output: seeds from all intervals
// parallelism: each block processes a read.
// limit:       summing up all the num_seeds from each intv
//              then allocating a memory and computing offsets
//              is serialized.

typedef cub::BlockRadixSort<uint64_t, SAL_SORT_BLOCKDIMX, SAL_SORT_ITEMS_PER_THREAD, int>
    SalBlockSort;
// CUB BlockScan for parallel offset/num_seeds prefix sum.
// Replaces the thread-0-only O(num_intvs) serial loop.
typedef cub::BlockScan<int, SAL_SORT_BLOCKDIMX>
    SalBlockScan;

__global__ void sa_lookup_kernel(
        const mem_opt_t *d_opt,
        const fmindex_t *devFmIndex,
        const bntseq_t *d_bns,
        const uint8_t *d_seq,
        int *d_seq_offset,
        smem_aux_t *d_aux,
        mem_seed_v *d_seq_seeds,	// output
        void *d_buffer_pools
        )
{
    int seqID = blockIdx.x;
    bwtintv_t *intvs = d_aux[seqID].mem.a;
    int num_intvs = d_aux[seqID].mem.n;

    // Sort omitted for non-GSS path: affects ~0.015% of chain compositions
    // and costs 12 KB static SMEM (s_reorder) + ~5 KB CUB sort TempStorage = 17 KB,
    // dropping theoretical occupancy from 75% to 58%. Not used for 76bp (GSS=OFF).
    // The GSS prepass (sa_lookup_prepass_emit_kernel) sorts intervals for 148bp reads.

    // scan TempStorage only (sort_tmp eliminated).
    __shared__ typename SalBlockScan::TempStorage sal_scan_tmp;

    // parallel prefix sum for offsets[] using CUB BlockScan.
    // s_offsets[SAL_SCAN_CAPACITY+1] covers all num_intvs up to 512 (the sort capacity),
    // eliminating the prior pool-allocation fallback for num_intvs > MAX_LEN_READ+1.
    __shared__ int s_offsets[SAL_SCAN_CAPACITY + 1];
    __shared__ mem_seed_t *seed_a;
    __shared__ float s_frac_rep;

    // --- Phase 1 (all threads): load per-interval seed counts, run BlockScan ---
    // Each thread loads SAL_SORT_ITEMS_PER_THREAD items (padded with 0 beyond num_intvs).
    int scan_in[SAL_SORT_ITEMS_PER_THREAD];
    for (int k = 0; k < SAL_SORT_ITEMS_PER_THREAD; k++) {
        int idx = threadIdx.x * SAL_SORT_ITEMS_PER_THREAD + k;
        if (idx < num_intvs) {
            bwtint_t x2 = intvs[idx].x[2];
            scan_in[k] = (int)(x2 > (bwtint_t)d_opt->max_occ ? d_opt->max_occ : x2);
        } else {
            scan_in[k] = 0;
        }
    }
    // ExclusiveSum across all 128 threads × 4 items; aggregate = total num_seeds.
    int scan_out[SAL_SORT_ITEMS_PER_THREAD];
    int total_seeds;
    SalBlockScan(sal_scan_tmp).ExclusiveSum(scan_in, scan_out, total_seeds);
    __syncthreads();

    // Write offsets[] to shared memory (each thread owns its slice).
    for (int k = 0; k < SAL_SORT_ITEMS_PER_THREAD; k++) {
        int idx = threadIdx.x * SAL_SORT_ITEMS_PER_THREAD + k;
        if (idx < num_intvs) {
            s_offsets[idx] = scan_out[k];
        }
    }
    if (threadIdx.x == 0) {
        s_offsets[num_intvs] = total_seeds;  // sentinel: total seed count
    }
    __syncthreads();

    // --- Phase 2 (thread 0): compute frac_rep (sequential interval-merge) and alloc ---
    // frac_rep uses a non-overlapping interval merge — sequential dependency, stays serial.
    if (threadIdx.x == 0) {
        void *buf = CUDAKernelSelectPool(d_buffer_pools, blockIdx.x % 32);
        int l_rep = 0, b = 0, e = 0;
        for (int k = 0; k < num_intvs; k++) {
            // Accumulate repetitive SMEM coverage (non-overlapping merge)
            if (intvs[k].x[2] > (bwtint_t)d_opt->max_occ) {
                int sb = M(intvs[k]);           // query begin
                int se = (int)N(intvs[k]) + 1;  // query end (exclusive)
                if (sb > e) { l_rep += e - b; b = sb; e = se; }
                else { e = e > se ? e : se; }
            }
        }
        l_rep += e - b;

        int l_seq = d_seq_offset[seqID + 1] - d_seq_offset[seqID];
        s_frac_rep = l_seq > 0 ? (float)l_rep / l_seq : 0.0f;

        seed_a = d_seq_seeds[seqID].a = (mem_seed_t*)CUDAKernelMalloc(
            buf, total_seeds * sizeof(mem_seed_t), 8);
        d_seq_seeds[seqID].n = total_seeds;
    }
    __shared__ int s_actual_count;
    if (threadIdx.x == 0) s_actual_count = 0;
    __syncthreads(); __syncwarp();

    float frac_rep = s_frac_rep;

    // collect seeds from each interval, filtering rid < 0 (cross-ref seeds)
    int intv_id;
    int intv_size, step_size;
    bwtintv_t *intv;
    for(intv_id = threadIdx.x; intv_id < num_intvs; intv_id += blockDim.x) {
        intv = &intvs[intv_id];
        intv_size = s_offsets[intv_id + 1] - s_offsets[intv_id];
        step_size = intv->x[2] > d_opt->max_occ ? intv->x[2] / d_opt->max_occ : 1;

        for(int j = 0; j < intv_size; j++) {
            mem_seed_t new_seed;
            bwtint_t k = intv->x[0] + step_size * j;
            if(k > *(devFmIndex->referenceLen)){
                continue;
            }
            int64_t rbeg;
            sa_lookup(devFmIndex, k, &rbeg);
            new_seed.rbeg = rbeg;
            new_seed.qbeg = M(*intv);
            new_seed.len = new_seed.score = LEN(*intv);
            new_seed.rid = bns_intv2rid_gpu(d_bns, new_seed.rbeg, new_seed.rbeg + new_seed.len);
            if (new_seed.rid < 0) continue; // skip seeds bridging reference sequences
            new_seed.frac_rep = frac_rep;
            int pos = atomicAdd(&s_actual_count, 1);
            seed_a[pos] = new_seed;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        d_seq_seeds[seqID].n = s_actual_count;
    }
}

// ============================================================================
// Gather-Sort-Scatter (GSS) for sa_lookup_kernel
// ============================================================================
//
// Overview:
//   1. sa_lookup_prepass_count_kernel: compute per-read total_seeds (int) into
//      d_per_read_total_seeds[].  Host then runs CUB ExclusiveSum to produce
//      d_read_base_offsets[] and batch_total_seeds.
//   2. sa_lookup_prepass_emit_kernel: emit (bwt_pos, seqID, meta) tuples into
//      flat arrays d_sal_keys[], d_sal_vals[], d_sal_meta[].
//   3. Host: cub::DeviceRadixSort::SortPairs(d_sal_keys → d_sal_keys_sorted,
//            d_sal_vals → d_sal_perm)
//   4. sa_lookup_gss_kernel: sorted keys → SA lookup → d_sal_rbeg[t]
//   5. sa_lookup_scatter_kernel: scatter (rbeg, meta, rid) → d_seq_seeds[seqID].a[]
//
// Dataset-conditional: gated at runtime by proc->use_gss and batch_total_seeds
// (see bwamem.cu). E.coli batches (SA table ~12 MB < L2 cache 40 MB) stay on
// the original path.

// ---- Kernel 1: count total seeds per read ----
// Launched <<<batch_size, SAL_SORT_BLOCKDIMX>>>.
// Uses same BlockScan logic as sa_lookup_kernel to compute total_seeds for each read,
// then thread 0 writes it to d_per_read_total_seeds[seqID].
__global__ void sa_lookup_prepass_count_kernel(
        const mem_opt_t  *d_opt,
        smem_aux_t       *d_aux,
        int              *d_per_read_total_seeds)   // out: [batch_size]
{
    int seqID = blockIdx.x;
    int num_intvs = d_aux[seqID].mem.n;
    if (num_intvs > SAL_SCAN_CAPACITY) num_intvs = SAL_SCAN_CAPACITY;  // guard: cap at scan capacity

    // Parallel prefix sum of per-interval seed counts (same as sa_lookup_kernel Phase 1).
    __shared__ typename SalBlockScan::TempStorage scan_tmp;

    int scan_in[SAL_SORT_ITEMS_PER_THREAD];
    for (int k = 0; k < SAL_SORT_ITEMS_PER_THREAD; k++) {
        int idx = threadIdx.x * SAL_SORT_ITEMS_PER_THREAD + k;
        if (idx < num_intvs) {
            bwtint_t x2 = d_aux[seqID].mem.a[idx].x[2];
            scan_in[k] = (int)(x2 > (bwtint_t)d_opt->max_occ ? d_opt->max_occ : x2);
        } else {
            scan_in[k] = 0;
        }
    }
    int scan_out[SAL_SORT_ITEMS_PER_THREAD];
    int total_seeds;
    SalBlockScan(scan_tmp).ExclusiveSum(scan_in, scan_out, total_seeds);
    __syncthreads();
    if (threadIdx.x == 0)
        d_per_read_total_seeds[seqID] = total_seeds;
}

// ---- Kernel 2: sort intervals, compute frac_rep, and emit (key=bwt_pos, val=seqID, meta) tuples ----
// Launched <<<batch_size, SAL_SORT_BLOCKDIMX>>>.
// Mirrors sa_lookup_kernel but emits to flat global arrays instead of allocating per-read.
// d_read_base_offsets[seqID] = exclusive prefix sum of d_per_read_total_seeds[].
__global__ void sa_lookup_prepass_emit_kernel(
        const mem_opt_t  *d_opt,
        const uint8_t    *d_seq,
        const int        *d_seq_offset,
        smem_aux_t       *d_aux,
        const int        *d_read_base_offsets,     // in: [batch_size+1]
        uint64_t         *d_sal_keys,              // out: [batch_total_seeds]
        int              *d_sal_vals,              // out: [batch_total_seeds] (seqID)
        sal_meta_t       *d_sal_meta)              // out: [batch_total_seeds]
{
    int seqID = blockIdx.x;
    bwtintv_t *intvs   = d_aux[seqID].mem.a;
    int num_intvs      = d_aux[seqID].mem.n;
    if (num_intvs > SAL_SCAN_CAPACITY) num_intvs = SAL_SCAN_CAPACITY;  // guard: cap at scan capacity
    int base_offset    = d_read_base_offsets[seqID];

    // Phase 0: sort intervals by info (= qbeg<<32|qend) — required for correct frac_rep merge.
    // Mirrors the sort step in sa_lookup_kernel; also writes sorted intervals back to global mem
    // so the original sa_lookup_kernel's sort (when GSS is not used) sees a sorted array.
    __shared__ union {
        typename SalBlockSort::TempStorage sort_tmp;
        typename SalBlockScan::TempStorage scan_tmp;
    } tmp;
    __shared__ bwtintv_t s_reorder[SAL_SCAN_CAPACITY];

    if (num_intvs > 1) {
        uint64_t keys[SAL_SORT_ITEMS_PER_THREAD];
        int      vals[SAL_SORT_ITEMS_PER_THREAD];
        for (int k = 0; k < SAL_SORT_ITEMS_PER_THREAD; k++) {
            int idx = threadIdx.x * SAL_SORT_ITEMS_PER_THREAD + k;
            keys[k] = (idx < num_intvs) ? intvs[idx].info : UINT64_MAX;
            vals[k] = idx;
        }
        __syncthreads();
        SalBlockSort(tmp.sort_tmp).Sort(keys, vals);
        __syncthreads();
        for (int k = 0; k < SAL_SORT_ITEMS_PER_THREAD; k++) {
            int new_rank = threadIdx.x * SAL_SORT_ITEMS_PER_THREAD + k;
            if (new_rank < num_intvs)
                s_reorder[new_rank] = intvs[vals[k]];
        }
        __syncthreads();
        for (int k = 0; k < SAL_SORT_ITEMS_PER_THREAD; k++) {
            int idx = threadIdx.x * SAL_SORT_ITEMS_PER_THREAD + k;
            if (idx < num_intvs)
                intvs[idx] = s_reorder[idx];
        }
    }
    __syncthreads();

    // Phase 1: parallel prefix sum to get per-interval output offsets (s_offsets[]).
    __shared__ int s_offsets[SAL_SCAN_CAPACITY + 1];

    int scan_in[SAL_SORT_ITEMS_PER_THREAD];
    for (int k = 0; k < SAL_SORT_ITEMS_PER_THREAD; k++) {
        int idx = threadIdx.x * SAL_SORT_ITEMS_PER_THREAD + k;
        if (idx < num_intvs) {
            bwtint_t x2 = intvs[idx].x[2];
            scan_in[k] = (int)(x2 > (bwtint_t)d_opt->max_occ ? d_opt->max_occ : x2);
        } else {
            scan_in[k] = 0;
        }
    }
    int scan_out[SAL_SORT_ITEMS_PER_THREAD];
    int total_seeds_local;
    SalBlockScan(tmp.scan_tmp).ExclusiveSum(scan_in, scan_out, total_seeds_local);
    __syncthreads();

    for (int k = 0; k < SAL_SORT_ITEMS_PER_THREAD; k++) {
        int idx = threadIdx.x * SAL_SORT_ITEMS_PER_THREAD + k;
        if (idx < num_intvs)
            s_offsets[idx] = scan_out[k];
    }
    if (threadIdx.x == 0)
        s_offsets[num_intvs] = total_seeds_local;
    __syncthreads();

    // Phase 2: compute frac_rep (thread 0 serial — true sequential dependency).
    // Intervals are now sorted by qbeg (info), so the non-overlapping merge is correct.
    __shared__ float s_frac_rep;
    if (threadIdx.x == 0) {
        int l_rep = 0, b = 0, e = 0;
        for (int k = 0; k < num_intvs; k++) {
            if (intvs[k].x[2] > (bwtint_t)d_opt->max_occ) {
                int sb = M(intvs[k]);
                int se = (int)N(intvs[k]) + 1;
                if (sb > e) { l_rep += e - b; b = sb; e = se; }
                else { e = e > se ? e : se; }
            }
        }
        l_rep += e - b;
        int l_seq = d_seq_offset[seqID + 1] - d_seq_offset[seqID];
        s_frac_rep = l_seq > 0 ? (float)l_rep / l_seq : 0.0f;
    }
    __syncthreads();
    float frac_rep = s_frac_rep;

    // Phase 3: emit (key, val=identity, meta) per slot.
    // d_sal_vals[slot] = slot (identity) so that after SortPairs,
    // d_sal_perm[j] = original slot of sorted rank j.
    // seqID is recovered in the scatter kernel via binary search on d_read_base_offsets.
    for (int intv_id = threadIdx.x; intv_id < num_intvs; intv_id += blockDim.x) {
        bwtintv_t *intv = &intvs[intv_id];
        int intv_size  = s_offsets[intv_id + 1] - s_offsets[intv_id];
        int step_size  = intv->x[2] > d_opt->max_occ ? (int)(intv->x[2] / d_opt->max_occ) : 1;
        int qbeg       = M(*intv);
        int len        = (int)LEN(*intv);
        for (int j = 0; j < intv_size; j++) {
            int global_slot = base_offset + s_offsets[intv_id] + j;
            uint64_t bwt_pos = intv->x[0] + (uint64_t)step_size * j;
            d_sal_keys[global_slot] = bwt_pos;
            d_sal_vals[global_slot] = global_slot;  // identity permutation
            d_sal_meta[global_slot] = {qbeg, len, frac_rep};
        }
    }
}

// ---- Kernel 3: sorted SA lookup ----
// Launched <<<ceil(N/SAL_SORT_BLOCKDIMX), SAL_SORT_BLOCKDIMX>>>.
// Thread t: read bwt_pos from sorted keys → sa_lookup → write rbeg to d_sal_rbeg[t].
// No shared memory needed — accesses to the SA are now in sorted BWT-position order.
__global__ void sa_lookup_gss_kernel(
        const fmindex_t  *devFmIndex,
        const uint64_t   *d_sal_keys_sorted,  // in: [N] sorted bwt positions
        int64_t          *d_sal_rbeg,          // out: [N] rbeg values (-1 if invalid)
        int               N)                   // total seeds
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    uint64_t bwt_pos = d_sal_keys_sorted[t];
    if (bwt_pos > *(devFmIndex->referenceLen)) {
        d_sal_rbeg[t] = -1;
        return;
    }
    int64_t rbeg;
    sa_lookup(devFmIndex, bwt_pos, &rbeg);
    d_sal_rbeg[t] = rbeg;
}

// ---- Kernel 3b: scatter metadata via permutation ----
// d_meta_sorted[i] = d_meta_orig[d_perm[i]]
__global__ void sal_meta_scatter_kernel(
        const sal_meta_t *d_meta_orig,
        sal_meta_t       *d_meta_sorted,
        const int        *d_perm,
        int               N)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    d_meta_sorted[t] = d_meta_orig[d_perm[t]];
}

// ---- Kernel 3c: allocate seed arrays and reset per-read actual counters ----
// Launched <<<ceil(batch_size/BLOCKDIM), BLOCKDIM>>>.
// Thread t (for seqID t): allocate d_seq_seeds[t].a from pool, set n = per_read_total,
// reset d_seq_actual_count[t] = 0.
__global__ void sal_alloc_seeds_kernel(
        const int    *d_per_read_total_seeds,  // [batch_size]
        mem_seed_v   *d_seq_seeds,              // out
        int          *d_seq_actual_count,       // out: reset to 0
        void         *d_buffer_pools,
        int           batch_size)
{
    int seqID = blockIdx.x * blockDim.x + threadIdx.x;
    if (seqID >= batch_size) return;
    int total = d_per_read_total_seeds[seqID];
    void *buf = CUDAKernelSelectPool(d_buffer_pools, seqID % 32);
    d_seq_seeds[seqID].a = (mem_seed_t*)CUDAKernelMalloc(
            buf, total * sizeof(mem_seed_t), 8);
    d_seq_seeds[seqID].n  = total;
    d_seq_actual_count[seqID] = 0;
}

// ---- Kernel 3d: write final .n counts from actual-count array ----
// After scatter, d_seq_seeds[seqID].n was set to total_seeds (pre-alloc estimate).
// Replace with actual count (excluding rid<0 seeds) from d_seq_actual_count[].
__global__ void sal_finalize_counts_kernel(
        const int  *d_seq_actual_count,  // [batch_size]
        mem_seed_v *d_seq_seeds,          // in/out
        int         batch_size)
{
    int seqID = blockIdx.x * blockDim.x + threadIdx.x;
    if (seqID >= batch_size) return;
    d_seq_seeds[seqID].n = d_seq_actual_count[seqID];
}

// ---- Kernel 4: scatter results back to d_seq_seeds ----
// Launched <<<ceil(N/SAL_SORT_BLOCKDIMX), SAL_SORT_BLOCKDIMX>>>.
// Thread t:
//   - orig_slot = d_sal_perm[t] (original flat global slot before sort)
//   - meta[t] = d_sal_meta_sorted[t] (already permuted by sal_meta_scatter_kernel)
//   - rbeg = d_sal_rbeg[t]; if rbeg < 0: skip
//   - seqID: binary search in d_read_base_offsets for orig_slot
//   - atomicAdd to per-read counter, write to d_seq_seeds[seqID].a[pos]
__global__ void sa_lookup_scatter_kernel(
        const fmindex_t  *devFmIndex,
        const bntseq_t   *d_bns,
        const int64_t    *d_sal_rbeg,              // [N]
        const int        *d_sal_perm,              // [N] original slot at sorted position t
        const sal_meta_t *d_sal_meta_sorted,       // [N] meta (permuted)
        int               N,
        const int        *d_read_base_offsets,     // [batch_size+1] per-read base offsets
        int               batch_size,
        mem_seed_v       *d_seq_seeds,             // out: seed vectors per read
        int              *d_seq_actual_count)      // out: per-read actual written count [batch_size]
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    int64_t rbeg = d_sal_rbeg[t];
    if (rbeg < 0) return;

    // Binary search to find seqID: d_read_base_offsets[seqID] <= orig_slot < d_read_base_offsets[seqID+1]
    int orig_slot = d_sal_perm[t];
    int lo = 0, hi = batch_size - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (d_read_base_offsets[mid] <= orig_slot) lo = mid;
        else hi = mid - 1;
    }
    int seqID = lo;

    sal_meta_t meta = d_sal_meta_sorted[t];

    mem_seed_t seed;
    seed.rbeg     = rbeg;
    seed.qbeg     = meta.qbeg;
    seed.len      = meta.len;
    seed.score    = meta.len;
    seed.frac_rep = meta.frac_rep;
    seed.rid      = bns_intv2rid_gpu(d_bns, rbeg, rbeg + meta.len);
    if (seed.rid < 0) return;

    int pos = atomicAdd(&d_seq_actual_count[seqID], 1);
    d_seq_seeds[seqID].a[pos] = seed;
}
