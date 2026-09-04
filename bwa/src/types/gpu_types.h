#ifndef GPU_TYPES_H_
#define GPU_TYPES_H_

#include "bwa.h"

typedef struct
{
	int seqID;			// read ID
	uint16_t chainID;	// index on the chain vector of the read
	uint16_t seedID	;	// index of seed on the chain
	uint16_t regID;		// index on the (mem_alnreg_t)regs.a vector
	// below are for SW extension
	uint8_t* read_left; 	// string of read on the left of seed
	uint8_t* ref_left;		// string of reference on the left of seed
	uint8_t* read_right; 	// string of read on the right of seed
	uint8_t* ref_right;		// string of reference on the right of seed
	uint16_t readlen_left; 	// length of read on the left of seed
	uint16_t reflen_left;	// length of reference on the left of seed
	uint16_t readlen_right; // length of read on the right of seed
	uint16_t reflen_right;	// length of reference on the right of seed
} seed_record_t;

typedef struct process_data_t {
    // constant pointers on device ( index, memory managment, user-options , etc. )
	mem_opt_t* d_opt;		// user-defined options
	bntseq_t* d_bns;
	uint8_t* d_pac;
	void* d_buffer_pools;	// buffer pools
	mem_pestat_t* d_pes; 	// paired-end stats
	mem_pestat_t* h_pes0;	// pes0 on host for paired-end stats
	kmers_bucket_t* d_kmerHashTab;
    // pointers that will change each batch (being swapped between transfer and process)
    fmindex_t *d_fmi;

        // reads on device
	int batch_size;			// number of reads on device
	int64_t n_processed;	// number of reads processed prior to this batch
	bseq1_t *d_seqs;		// reads
    char *d_seq_name_ptr, *d_seq_comment_ptr, *d_seq_seq_ptr, *d_seq_qual_ptr, *d_seq_sam_ptr;  // name, comment, seq, qual, sam output
	int *d_seq_sam_size;	// length of sam on device, also used for all threads to atomic write SAM
	    // pre-allocated reads on host
    bseq1_t *h_seqs;		// reads
    char *h_seq_name_ptr, *h_seq_comment_ptr, *h_seq_seq_ptr, *h_seq_qual_ptr, *h_seq_sam_ptr;  // name, comment, seq, qual, sam output

        // intermediate data on device
	seed_record_t *d_seed_records; 	// global records of seeds, a big chunk of memory
	int *d_Nseeds;			// device pointer (host-mapped) — total seeds for SW extension
	int *d_Nalns;			// device pointer (host-mapped) — total alns for traceback
	int *h_Nseeds;			// host-mapped pinned pointer for d_Nseeds
	int *h_Nalns;			// host-mapped pinned pointer for d_Nalns
	void *d_lifetime_arena;	// single backing alloc for d_aux+d_seq_seeds (early) / d_seed_records (late)
	smem_aux_t* d_aux;		// collections of SA intervals, vector of size nseqs
	mem_seed_v* d_seq_seeds;// seeds array for each read
	mem_chain_v *d_chains;	// chain vectors of size nseqs
	mem_alnreg_v *d_regs;	// alignment info vectors, size nseqs

	        // arrays for sorting, each has length = batch_size
	int *d_sortkeys_in;
	int *d_seqIDs_in;
	int *d_sortkeys_out;
	int *d_seqIDs_out;
	int n_sortkeys;

    // pointers to CUDA stream, using generic pointers for compatibility with C
    void *CUDA_stream;   // process stream
    int gpu_no;
    int batch_no;

    uint64_t batch_offset;

    int *d_offsets;
    int *d_rids;
    int64_t *d_positions;
    int *d_ncigars;
    uint32_t *d_cigars;
    int *d_flags;   // SAM FLAG per alignment
    int *d_mapqs;   // MAPQ per alignment

	mem_aln_v * d_alns;		// alignment vectors, size nseqs

    int *d_total_alns_num;  // total #alns for this batch of reads.
    // flattened structure for final alignment results.
    // produced in 2-pass to determine #alns per read first.
    // cigars are produced in 3rd pass after determining the lengths first.
    int *d_alns_num;        // # alns per read. ID: readID
    int *d_alns_offset;     // alnID offset per read. ID: readID

    int *d_alns_rid;        // rid per aln. ID: alnID
    int64_t *d_alns_pos;        // pos per aln. ID: alnID

    int *d_total_cigar_len;     // total cigar len for this batch.
    int *d_alns_cigar_len;        // cigar len per aln. ID: alnID
    uint32_t *d_alns_cigar_offset;     // cigar offset per aln. ID: alnID
    int *d_alns_cigar;     // cigars per aln. ID: cigar offset & len.

    uint8_t *h_seq;
    int *h_seq_offset;

    uint8_t *d_seq;
    int *d_seq_offset;

    int *d_rid;
    int *d_aln_offsets;
    uint64_t *d_pos;
    int *d_chunk_aln_count;

    size_t intermediate_bytes; // total device VRAM allocated in device_alloc (excludes pool and index)

    void  *d_cub_temp;       // preallocated CUB temp storage (avoids per-batch cudaMalloc/cudaFree)
    size_t d_cub_temp_bytes; // size of d_cub_temp in bytes

    // Gather-Sort-Scatter (GSS) buffers for sa_lookup_kernel.
    // Sized at SAL_GSS_MAX_SEEDS = max_reads * SAL_GSS_MAX_SEEDS_PER_READ.
    // All arrays are batch-scoped: written in prepass, sorted, read in GSS+scatter.
    bool    use_gss;                // true if SA table >> L2 cache (set in memcpy_index)
    int     gss_max_seeds;          // capacity of the arrays below (set in device_alloc)
    int    *d_per_read_total_seeds; // [max_reads] per-read seed count from prepass
    int    *d_read_base_offsets;    // [max_reads+1] exclusive prefix sum of above
    uint64_t *d_sal_keys;           // [gss_max_seeds] bwt positions (input to sort)
    uint64_t *d_sal_keys_sorted;    // [gss_max_seeds] bwt positions (sorted output)
    int      *d_sal_vals;           // [gss_max_seeds] seqID per slot (input to sort)
    int      *d_sal_perm;           // [gss_max_seeds] sorted seqID (sort output / perm)
    void     *d_sal_meta_buf;       // raw buffer: 2 × gss_max_seeds × 12 B (sal_meta_t)
    // d_sal_meta and d_sal_meta_sorted are carved from d_sal_meta_buf:
    //   d_sal_meta        = (sal_meta_t*)d_sal_meta_buf
    //   d_sal_meta_sorted = (sal_meta_t*)d_sal_meta_buf + gss_max_seeds
    int64_t  *d_sal_rbeg;           // [gss_max_seeds] rbeg results from GSS kernel
    int      *d_seq_actual_count;   // [max_reads] per-read actual written count
    void     *d_gss_cub_temp;       // CUB temp storage for DeviceRadixSort+DeviceScan
    size_t    d_gss_cub_temp_bytes; // size of d_gss_cub_temp

    // pinned host output buffers for async D2H transfers.
    // cudaMemcpyAsync to pageable memory is synchronous; pinned memory enables true DMA.
    int      *h_out_offsets;   // [MAX_BATCH_SIZE+1] sync-copied (needed before async starts)
    int      *h_out_rids;      // [MAX_ALN_CNT]
    int64_t  *h_out_positions; // [MAX_ALN_CNT]
    int      *h_out_ncigars;   // [MAX_ALN_CNT]
    uint32_t *h_out_cigars;    // [MAX_ALN_CNT * MAX_N_CIGAR]
    int      *h_out_flags;     // [MAX_ALN_CNT]
    int      *h_out_mapqs;     // [MAX_ALN_CNT]
} process_data_t;

#endif
