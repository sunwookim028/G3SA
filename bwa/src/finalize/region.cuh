#ifndef _REGION_CUH
#define _REGION_CUH

#include "gpu_types.h"
#include <cub/cub.cuh>

#define MAX_N_ALN 3072	// max number of alignments allowed per read

/* ------------------------------------------------------------------ */
/*  sort_regions shared memory layout                                  */
/* ------------------------------------------------------------------ */
#define SORT_REGIONS_BLOCK 256
#define SORT_REGIONS_ITEMS 12       /* 256 * 12 = 3072 = MAX_N_ALN */

typedef cub::BlockRadixSort<uint64_t, SORT_REGIONS_BLOCK, SORT_REGIONS_ITEMS, int>
    BlockRegionSort;

/* CUB BlockScan for parallel compaction (Steps 4 & 6 in patch_regions).
 * Items-per-thread = SORT_REGIONS_ITEMS so one call scans all MAX_N_ALN elements.
 * TempStorage is ~64 bytes (warp-scan algorithm on sm_86). */
typedef cub::BlockScan<int, SORT_REGIONS_BLOCK>
    BlockCompact;

struct SortRegionsSM {
    typename BlockRegionSort::TempStorage sort_tmp;
    uint16_t new_i[MAX_N_ALN];
    int      reduce_SM[8];
};

#define SORT_REGIONS_SM_BYTES (sizeof(SortRegionsSM))

/* ------------------------------------------------------------------ */
/*  filter_contained_regions dedicated sort layout                    */
/*                                                                      */
/*  filter_contained_regions caps n_use = min(n, 256), so it never     */
/*  sorts more than 256 items.  Using 1 item/thread vs 12 items/thread  */
/*  eliminates two 12-element register arrays (~22 regs freed), likely  */
/*  dropping total registers from 80 → ~58 (< 64 threshold for 4       */
/*  blocks/SM at 256 threads on sm_86).                                 */
/*                                                                      */
/*  FcrSM is ~4 KB vs SortRegionsSM ~30 KB.                             */
/* ------------------------------------------------------------------ */
#define FCR_SORT_ITEMS 1   /* 1 item/thread × 256 threads = 256 items max */

typedef cub::BlockRadixSort<uint64_t, SORT_REGIONS_BLOCK, FCR_SORT_ITEMS, int>
    FcrBlockSort;

struct FcrSM {
    typename FcrBlockSort::TempStorage sort_tmp;  /* ~3.3 KB */
    uint16_t new_i[256];                          /* 512 B  */
    int      reduce_SM[8];                        /* 32 B   */
};

#define FCR_SM_BYTES (sizeof(FcrSM))

/* ------------------------------------------------------------------ */
/*  patch_regions shared memory layout                                */
/*                                                                      */
/*  Layout (placed back-to-back inside the dynamic SM region):          */
/*    SortRegionsSM  (CUB temp + new_i[] + reduce_SM[])                  */
/*    PatchSM        (per-outer-iter scratch:                             */
/*                      j_min/j_kill_max/has_merge broadcast slots,      */
/*                      new_a_ptr pool slot,                              */
/*                      class_code[])                                     */
/*                                                                      */
/*  Phase-A classification codes (`class_code[j]`, applied by Phase B):  */
/*    0 = idle (dead, predicate failed, no overlap, or deferred merge     */
/*        candidate — merge attempts are only retried serially by Phase B */
/*        once any_merge is already true; see patch_regions for why the   */
/*        initial-pass attempt is skipped)                                */
/*    1 = dedup high-overlap, kill-i  (p->score <  q->score)             */
/*    2 = dedup high-overlap, kill-j  (p->score >= q->score)             */
/*                                                                      */
/*  PatchSM no longer carries a merge_score[] array: the Phase-A          */
/*  forward-strand merge fast path (which used to populate it) is        */
/*  disabled by default, shrinking PatchSM from 15 KB → 3 KB.  Combined   */
/*  with SortRegionsSM (30 KB): 45 KB → 33 KB per block.  At 33 KB:       */
/*  100 KB / 33 KB = 3.02 → 3 blocks/SM (was 2, SM-limited).              */
/* ------------------------------------------------------------------ */
struct PatchSM {
    int  j_min;
    int  j_kill_max;
    int  has_merge;
    int  _pad;             /* keep struct size aligned to 8-byte ptr */
    void *new_a_ptr;       /* opaque to host; sized as a pointer */
    uint8_t  class_code [MAX_N_ALN];  /* 3072 B  */
};

#define PATCH_REGIONS_SM_BYTES (sizeof(SortRegionsSM) + sizeof(PatchSM))

/* GPU port of bwa-mem2's post-extension seed containment check.
   Purges regions whose seeds are fully contained in higher-scoring regions.
   Launch: gridDim.x = ceil(n_seqs/WARPSIZE), blockDim.x = WARPSIZE.
 */
__global__ void filter_contained_regions(
        const mem_opt_t *d_opt,
        mem_chain_v *d_chains,
        mem_alnreg_v *d_regs,
        seed_record_t *d_seed_records,
        int *d_Nseeds,
        int *d_seq_offset,
        int batch_size
        );

/* GPU port of bwa-mem2's mem_sort_dedup_patch.
   Merges partially-overlapping colinear regions via global alignment,
   deduplicates highly-overlapping regions, and compacts the result.
   Launch: <<<batch_size, SORT_REGIONS_BLOCK, PATCH_REGIONS_SM_BYTES>>>
   — one block per read; sorts use parallel CUB radix sort for large n,
   dedup is block-cooperative, merge falls back to serial thread-0 work
   (sequential mutation of p across --j gates later overlap decisions).
 */
__global__ void patch_regions(
        const mem_opt_t  *d_opt,
        const bntseq_t   *d_bns,
        const uint8_t    *d_pac,
        uint8_t          *d_seq,
        const int        *d_seq_offset,
        mem_alnreg_v     *d_regs,
        void             *d_buffer_pools,
        int               batch_size
        );

/* Fused filter_regions + apply_score_filter.
 * Pass 1 (256 threads): set is_alt flags.
 * Pass 2 (thread 0):    compact out score < opt->T regions.
 * Must run BEFORE sort_regions so is_alt is set for the two-pass sort.
 * Launch: <<<batch_size, 256, 0, stream>>>
 */
__global__ void filter_and_score_threshold(
        const mem_opt_t *d_opt,
        const bntseq_t  *d_bns,
        mem_alnreg_v    *d_regs,
        int64_t          batch_offset);

/*
 * Full port of bwa-mem2's mem_mark_primary_se (two-pass alt/non-alt).
 * Handles the sort comparator, stale field resets, and two-pass
 * re-sort/re-mark for mixed alt/non-alt regions.
 * batch_offset = n_processed at the time of the call (for hash tiebreaker).
 */
__global__ void sort_regions(
        const mem_opt_t *d_opt,
        mem_alnreg_v *d_regs,
        int n_seqs,
        int64_t batch_offset,
        void *d_buffer_pools
        );

#endif
