#ifndef _EXTEND_CUH
#define _EXTEND_CUH

#include "gpu_types.h"

/* preprocessing 1 for SW extension
   count the number of seeds for each read and write to global records, allocate output regs vector
 */
__global__ void sw_seed_prep(
        mem_chain_v *d_chains,
        mem_alnreg_v *d_regs,
        seed_record_t *d_seed_records,
        int *d_Nseeds,	// total seed count across all reads
        int n_seqs,	// number of reads
        void* d_buffer_pools
        );

/* Fused extend_pair_generate + local_extend.
 * Eliminates 80 B/seed global round-trip through d_seed_records.
 * d_seed_records is read (metadata fields) but NOT written.
 * Launch: <<<num_seeds_to_extend, WARPSIZE>>>
 */
__global__ void extend_and_sw(
        const mem_opt_t *d_opt,
        bntseq_t *d_bns,
        uint8_t *d_pac,
        uint8_t *d_seq,
        int *d_seq_offset,
        mem_chain_v *d_chains,
        mem_alnreg_v *d_regs,
        seed_record_t *d_seed_records,
        int *d_Nseeds,
        int n_seqs,
        void* d_buffer_pools
        );

#endif
