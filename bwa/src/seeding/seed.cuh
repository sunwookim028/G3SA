#ifndef _SEED_CUH
#define _SEED_CUH
#include "bwa.h"
#include "bwt.h"
#include <cub/cub.cuh>

// Block dimension for sa_lookup_kernel (must match launch config in bwamem.cu)
#define SAL_SORT_BLOCKDIMX        128
// Items per thread for CUB BlockRadixSort/BlockScan: 128 * 4 = 512 >= MAX_LEN_READ (320)
#define SAL_SORT_ITEMS_PER_THREAD 4
// Total capacity: must cover max num_intvs per read
#define SAL_SCAN_CAPACITY (SAL_SORT_BLOCKDIMX * SAL_SORT_ITEMS_PER_THREAD)
// Minimum average read length (bp) for the GSS prepass to be worthwhile.
// For short reads (76 bp) the GSS prepass overhead exceeds the savings.
// For longer reads (148 bp) GSS is worthwhile (more seeds → better L2 hit rate).
#define SAL_GSS_MIN_READ_LEN      100

__global__ void preseed_and_filter(
        const fmindex_t  *devFmIndex,
        const mem_opt_t *d_opt, 
        const uint8_t *d_seq,
        int *d_seq_offset,
        smem_aux_t *d_aux, 			// aux output
        kmers_bucket_t *d_kmerHashTab,
        void *d_buffer_pools);

__global__ void reseedV2(
        const fmindex_t *devFmIndex,
        const mem_opt_t *d_opt,
        uint8_t *d_seq,
        int *d_seq_offset,
        smem_aux_t *d_aux, 			// aux output
        kmers_bucket_t *d_kmerHashTab,
        void * d_buffer_pools,
        int num_reads
        );

// calculate necessary SMEM3
__global__ void reseedLastRound(
        const fmindex_t *devFmIndex,
        const mem_opt_t *d_opt,
        uint8_t *d_seq,
        int *d_seq_offset,
        smem_aux_t *d_aux,
        kmers_bucket_t *d_kmerHashTab,
        int numReads
        );


// input: mem intervals
// output: seeds from all intervals
// parallelism: each block processes a read.
__global__ void sa_lookup_kernel(
        const mem_opt_t *d_opt,
        const fmindex_t *devFmIndex,
        const bntseq_t *d_bns,
        const uint8_t *d_seq,
        int *d_seq_offset,
        smem_aux_t *d_aux,
        mem_seed_v *d_seq_seeds,	// output
        void *d_buffer_pools
        );

// GSS (Gather-Sort-Scatter) kernels for sa_lookup_kernel.
// Per-seed metadata emitted by the pre-pass and consumed by the scatter kernel.
struct sal_meta_t {
    int   qbeg;
    int   len;
    float frac_rep;
};

__global__ void sa_lookup_prepass_count_kernel(
        const mem_opt_t  *d_opt,
        smem_aux_t       *d_aux,
        int              *d_per_read_total_seeds);

__global__ void sa_lookup_prepass_emit_kernel(
        const mem_opt_t  *d_opt,
        const uint8_t    *d_seq,
        const int        *d_seq_offset,
        smem_aux_t       *d_aux,
        const int        *d_read_base_offsets,
        uint64_t         *d_sal_keys,
        int              *d_sal_vals,
        sal_meta_t       *d_sal_meta);

__global__ void sa_lookup_gss_kernel(
        const fmindex_t  *devFmIndex,
        const uint64_t   *d_sal_keys_sorted,
        int64_t          *d_sal_rbeg,
        int               N);

__global__ void sal_meta_scatter_kernel(
        const sal_meta_t *d_meta_orig,
        sal_meta_t       *d_meta_sorted,
        const int        *d_perm,
        int               N);

__global__ void sal_alloc_seeds_kernel(
        const int    *d_per_read_total_seeds,
        mem_seed_v   *d_seq_seeds,
        int          *d_seq_actual_count,
        void         *d_buffer_pools,
        int           batch_size);

__global__ void sal_finalize_counts_kernel(
        const int  *d_seq_actual_count,
        mem_seed_v *d_seq_seeds,
        int         batch_size);

__global__ void sa_lookup_scatter_kernel(
        const fmindex_t  *devFmIndex,
        const bntseq_t   *d_bns,
        const int64_t    *d_sal_rbeg,
        const int        *d_sal_perm,
        const sal_meta_t *d_sal_meta_sorted,
        int               N,
        const int        *d_read_base_offsets,
        int               batch_size,
        mem_seed_v       *d_seq_seeds,
        int              *d_seq_actual_count);

#endif
