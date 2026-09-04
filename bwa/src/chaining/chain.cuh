#ifndef _CHAIN_CUH
#define _CHAIN_CUH

/* see chain.cu for GPL-3.0 attribution notice */

#include "gpu_types.h"

/* Constants for seed sorting */
#define SORTSEEDSHIGH_MAX_NSEEDS 	2048
#define SORTSEEDSHIGH_NKEYS_THREAD	16
#define SORTSEEDSHIGH_BLOCKDIMX		128
#define SORTSEEDSLOW_MAX_NSEEDS 	64
#define SORTSEEDSLOW_NKEYS_THREAD	2
#define SORTSEEDSLOW_BLOCKDIMX		32

/* Constants for chaining */
#define SEEDS_PER_CHAIN 4
#define MEM_SHORT_EXT 50
#define MEM_SHORT_LEN 200
#define MEM_HSP_COEF 1.1f
#define MEM_MINSC_COEF 5.5f
#define MEM_SEEDSW_COEF 0.05f

/* Constants for chain sorting */
#define MAX_N_CHAIN 		4096
#define NKEYS_EACH_THREAD	16
#define SORTCHAIN_BLOCKDIMX	128

/* Constants for parallel chaining (chain_seeds_parallel) */
#define CHAIN_PARALLEL_BLOCKDIMX  256
#define CHAIN_PARALLEL_MAX_NSEEDS SORTSEEDSHIGH_MAX_NSEEDS   /* 2048 */

/* Constants for chain filtering */
#define CHAIN_FLT_BLOCKSIZE 256
#define chn_beg(ch) ((ch).seeds->qbeg)
#define chn_end(ch) ((ch).seeds[(ch).n-1].qbeg + (ch).seeds[(ch).n-1].len)
#define GET_KEPT(i) (chn_info_SM[i]&0x3) 		// last 2 bits
#define SET_KEPT(i, val) (chn_info_SM[i]&=0b11111100)|=val
#define GET_IS_ALT(i) ((chn_info_SM[i]&0x4)>>2) 	// 3rd bit
#define SET_IS_ALT(i, val) (chn_info_SM[i]&=0b11111011)|=(val<<2)

/* Seed sorting kernels */

/* for each read, sort seeds by len ASC, then by qbeg ASC
   use cub::blockRadixSort
 */
// process reads who have less seeds
__global__ void sort_seeds_low(
        mem_seed_v *d_seq_seeds,
        void *d_buffer_pools
        );

// process reads who have more seeds
__global__ void sort_seeds_high(
        mem_seed_v *d_seq_seeds,
        void *d_buffer_pools
        );

/* Chaining kernels */

/* Block-per-read parallel chaining (production chaining kernel).
 * One CUDA block per read; all threads cooperate on predecessor search. */
__global__ void chain_seeds_parallel(
        int batch_size,
        const mem_opt_t *d_opt,
        const bntseq_t *d_bns,
        const uint8_t *d_seq,
        int *d_seq_offset,
        mem_seed_v *d_seq_seeds,
        mem_chain_v *d_chains,
        void *d_buffer_pools
        );

/* sort chains of each read by weight */
__global__ void sort_chains_by_weight(mem_chain_v* d_chains, void* d_buffer_pools);

/* Chain filtering kernels */

/* each block takes care of 1 read, do pairwise comparison of chains
   max number of chain is MAX_N_CHAIN
 */
__global__ void filter_chains(
        const mem_opt_t *opt,
        mem_chain_v *d_chains, 	// input and output
        void* d_buffer_pools);

__global__ void filter_chained_seeds(
        const mem_opt_t *d_opt, const bntseq_t *d_bns, const uint8_t *d_pac,
        const uint8_t *d_seq, const int *d_seq_offset,
        mem_chain_v *d_chains, 	// input and output
        int n,		// number of seqs
        void* d_buffer_pools
        );

#endif
