#ifndef _FINALIZE_CUH
#define _FINALIZE_CUH

#include "gpu_types.h"

/* See finalize.cu for GPL-3.0 attribution (portions adapted from minhhpham/bwa). */

#define GLOBALSW_BANDWITH_CUTOFF 500

/* prepare ref sequence for global SW
   allocate mem_aln_v array for each read
 */
__global__ void finalize_prep1(
        int batch_size,
        mem_alnreg_v* d_regs,
        mem_aln_v * d_alns,
        seed_record_t *d_seed_records,
        int *d_Nseeds,	// running total seed/aln-record count across all reads (atomic accumulator)
        void* d_buffer_pools);

/* run at aln level
   prepare seqs for SW global
 */
__global__ void finalize_prep2(
        const mem_opt_t *d_opt,
        uint8_t *d_seq,
        int *d_seq_offset,
        const uint8_t *d_pac,
        const bntseq_t *d_bns,
        mem_alnreg_v* d_regs,
        mem_aln_v * d_alns,
        seed_record_t *d_seed_records,
        int Nseeds,
        int *d_sortkeys_in,	// for sorting
        int *d_seqIDs_in,	// for sorting
        void* d_buffer_pools);

/* Fused traceback + finalize kernel */
__global__ void traceback_and_finalize(
        const mem_opt_t *d_opt,
        const bntseq_t *d_bns,
        const uint8_t *d_seq,
        int *d_seq_offset,
        mem_alnreg_v *d_regs,
        mem_aln_v *d_alns,
        seed_record_t *d_seed_records,
        int Nseeds,
        int *d_seqIDs_out,
        void *d_buffer_pools,
        int *d_offsets,
        int *d_rids,
        int64_t *d_positions,
        int *d_ncigars,
        uint32_t *d_cigars,
        int *d_flags,
        int *d_mapqs
        );

#endif
