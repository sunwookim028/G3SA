#include "gpu_types.h"
#include "cuda_wrapper.h"
#include "gmem_alloc.cuh"
#include "macro.h"
#include "errchk.cuh"
#include "timer.h"
#include "seed.cuh"
#include <cub/cub.cuh>
#include <iostream>


/* transfer index data */
static void transferIndex(
	const bntseq_t *bns,
	const uint8_t *pac,
	const kmers_bucket_t *kmerHashTab,
	process_data_t *process_instance,
    unsigned long long *allocated_size)
{
		/* CUDA GLOBAL MEMORY ALLOCATION AND TRANSFER */

	unsigned long long total_size = bns->n_seqs*sizeof(bntann1_t) +\
                                    bns->n_holes*sizeof(bntamb1_t) +\
                                    bns->l_pac*sizeof(uint8_t);

	// BNS
	// First create h_bns as a copy of bns on host
	// Then allocate its member pointers on device and copy data over
	// Then copy h_bns to d_bns
	uint32_t i, size;			// loop index and length of strings
	bntseq_t* h_bns;			// host copy to modify pointers
	h_bns = (bntseq_t*)malloc(sizeof(bntseq_t));
	memcpy(h_bns, bns, sizeof(bntseq_t));
	h_bns->anns = (bntann1_t*)malloc(bns->n_seqs*sizeof(bntann1_t));
	memcpy(h_bns->ambs, bns->ambs, bns->n_holes*sizeof(bntamb1_t));
	h_bns->ambs = (bntamb1_t*)malloc(bns->n_holes*sizeof(bntamb1_t));
	memcpy(h_bns->anns, bns->anns, bns->n_seqs*sizeof(bntann1_t));

		// allocate anns.name
	for (i=0; i<bns->n_seqs; i++){
		size = strlen(bns->anns[i].name);
		// allocate this name and copy to device
		cudaMalloc((void**)&(h_bns->anns[i].name), size+1); 			// +1 for "\0"
		cudaMemcpy(h_bns->anns[i].name, bns->anns[i].name, size+1, cudaMemcpyHostToDevice);
	}
	// allocate anns.anno
	for (i=0; i<bns->n_seqs; i++){
		size = strlen(bns->anns[i].anno);
		// allocate this name and copy to device
		cudaMalloc((void**)&(h_bns->anns[i].anno), size+1); 			// +1 for "\0"
		cudaMemcpy(h_bns->anns[i].anno, bns->anns[i].anno, size+1, cudaMemcpyHostToDevice);
	}
		// now h_bns->anns has pointers of name and anno on device
		// allocate anns on device and copy data from h_bns->anns to device
	bntann1_t* temp_d_anns;
	cudaMalloc((void**)&temp_d_anns, bns->n_seqs*sizeof(bntann1_t));
	cudaMemcpy(temp_d_anns, h_bns->anns, bns->n_seqs*sizeof(bntann1_t), cudaMemcpyHostToDevice);
		// now assign this pointer to h_bns->anns
	h_bns->anns = temp_d_anns;

		// allocate bns->ambs on device and copy data to device
	cudaMalloc((void**)&h_bns->ambs, bns->n_holes*sizeof(bntamb1_t));
	cudaMemcpy(h_bns->ambs, bns->ambs, bns->n_holes*sizeof(bntamb1_t), cudaMemcpyHostToDevice);

		// finally allocate d_bns and copy from h_bns
	bntseq_t* d_bns;
	cudaMalloc((void**)&d_bns, sizeof(bntseq_t));
	cudaMemcpy(d_bns, h_bns, sizeof(bntseq_t), cudaMemcpyHostToDevice);

	// PAC
	uint8_t* d_pac ;
	cudaMalloc((void**)&d_pac, bns->l_pac/4*sizeof(uint8_t)); 		// l_pac is length of ref seq
	cudaMemcpy(d_pac, pac, bns->l_pac/4*sizeof(uint8_t), cudaMemcpyHostToDevice); 		// divide by 4 because 2-bit encoding

	// K-MER HASH TABLE
	kmers_bucket_t* d_kmerHashTab ;
	cudaMalloc((void**)&d_kmerHashTab, pow4(KMER_K)*sizeof(kmers_bucket_t)); 		// l_pac is length of ref seq
	cudaMemcpy(d_kmerHashTab, kmerHashTab, pow4(KMER_K)*sizeof(kmers_bucket_t), cudaMemcpyHostToDevice); 		// divide by 4 because 2-bit encoding


	// output
	process_instance->d_bns = d_bns;
	process_instance->d_pac = d_pac;
	process_instance->d_kmerHashTab = d_kmerHashTab;
    long long kmer_size = (long long)pow4(KMER_K) * sizeof(kmers_bucket_t);
    long long pac_size  = (long long)(bns->l_pac / 4) * sizeof(uint8_t);
    std::cerr << "* bns+pac index " << total_size / MB_SIZE << " MB"
              << "  pac " << pac_size / MB_SIZE << " MB"
              << "  kmer " << kmer_size / MB_SIZE << " MB\n";
}

/* transfer user-defined optinos */
static void transferOptions(
	const mem_opt_t *opt, 
	mem_pestat_t *pes0,
	process_data_t *process_instance,
    unsigned long long *allocated_size)
{
	// matching and mapping options (opt)
	mem_opt_t* d_opt;
	cudaMalloc((void**)&d_opt, sizeof(mem_opt_t));
	cudaMemcpy(d_opt, opt, sizeof(mem_opt_t), cudaMemcpyHostToDevice);

	// paired-end stats: only allocate on device
	mem_pestat_t* d_pes;
	if (opt->flag&MEM_F_PE){
		cudaMalloc((void**)&d_pes, 4*sizeof(mem_pestat_t));
	}

	// output
	process_instance->d_opt = d_opt;
	process_instance->d_pes = d_pes;
	process_instance->h_pes0 = pes0;
}

/* transfer index data */
static void transferFmIndex(
        process_data_t *process_instance,
        const fmindex_t *idx,
    unsigned long long *allocated_size)
{
    /**
     * Reference data to transfer:
     *      count, count2
     *      cpOcc, cpOcc2
     *      oneHot, sentinelIndex, firstBase
     */
    fmindex_t hostFmIndex;

    long long size = 0;

    // FOR BWT-2
    uint64_t *d_one_hot;
    int sizeOneHot = 64 * sizeof(uint64_t);
    size += sizeOneHot;
    CUDA_CHECK(cudaMalloc((void**)&d_one_hot, sizeOneHot));
    CUDA_CHECK(cudaMemcpy(d_one_hot, idx->oneHot, sizeOneHot, cudaMemcpyHostToDevice));

    CP_OCC *d_cp_occ;
    int64_t cp_occ_size = idx->cpOccSize;
    size += cp_occ_size*sizeof(CP_OCC);
    CUDA_CHECK(cudaMalloc((void**)&d_cp_occ, cp_occ_size*sizeof(CP_OCC)));
    CUDA_CHECK(cudaMemcpy(d_cp_occ, idx->cpOcc, cp_occ_size*sizeof(CP_OCC), cudaMemcpyHostToDevice));

    int64_t *d_count;
    int sizeCount = sizeof(int64_t) * 5;
    size += sizeCount;
    CUDA_CHECK(cudaMalloc((void**)&d_count, sizeCount));
    CUDA_CHECK(cudaMemcpy(d_count, idx->count, sizeCount, cudaMemcpyHostToDevice));


    CP_OCC2 *d_cp_occ2;
    size += cp_occ_size*sizeof(CP_OCC2);
    CUDA_CHECK(cudaMalloc((void**)&d_cp_occ2, cp_occ_size*sizeof(CP_OCC2)));
    CUDA_CHECK(cudaMemcpy(d_cp_occ2, idx->cpOcc2, cp_occ_size*sizeof(CP_OCC2), cudaMemcpyHostToDevice));

    int64_t *d_count2;
    int sizeCount2 = sizeof(int64_t) * 17;
    size += sizeCount2;
    CUDA_CHECK(cudaMalloc((void**)&d_count2, sizeCount2));
    CUDA_CHECK(cudaMemcpy(d_count2, idx->count2, sizeCount2, cudaMemcpyHostToDevice));

    uint8_t *d_first_base;
    CUDA_CHECK(cudaMalloc((void**)&d_first_base, sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_first_base, idx->firstBase, sizeof(uint8_t), cudaMemcpyHostToDevice));

    int64_t *deviceSentinelIndex;
    CUDA_CHECK(cudaMalloc((void**)&deviceSentinelIndex, sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(deviceSentinelIndex, idx->sentinelIndex, sizeof(int64_t), cudaMemcpyHostToDevice));

    // SA lookup fields (suffixArrayMsByte, suffixArrayLsWord, referenceLen, packedBwt)
    int64_t refLen = *(idx->referenceLen);
    int64_t sa_num_entries = (refLen >> SA_COMPX) + 1;

    int64_t *d_referenceLen;
    CUDA_CHECK(cudaMalloc((void**)&d_referenceLen, sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_referenceLen, idx->referenceLen, sizeof(int64_t), cudaMemcpyHostToDevice));
    size += sizeof(int64_t);

    int8_t *d_suffixArrayMsByte;
    size_t saMsByteSize = sa_num_entries * sizeof(int8_t);
    CUDA_CHECK(cudaMalloc((void**)&d_suffixArrayMsByte, saMsByteSize));
    CUDA_CHECK(cudaMemcpy(d_suffixArrayMsByte, idx->suffixArrayMsByte, saMsByteSize, cudaMemcpyHostToDevice));
    size += saMsByteSize;

    uint32_t *d_suffixArrayLsWord;
    size_t saLsWordSize = sa_num_entries * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc((void**)&d_suffixArrayLsWord, saLsWordSize));
    CUDA_CHECK(cudaMemcpy(d_suffixArrayLsWord, idx->suffixArrayLsWord, saLsWordSize, cudaMemcpyHostToDevice));
    size += saLsWordSize;

    uint8_t *d_packedBwt;
    int64_t refLenAligned = ((refLen + CP_BLOCK_SIZE - 1) / CP_BLOCK_SIZE) * CP_BLOCK_SIZE;
    size_t packedBwtSize = refLenAligned / 4 * sizeof(uint8_t);
    CUDA_CHECK(cudaMalloc((void**)&d_packedBwt, packedBwtSize));
    CUDA_CHECK(cudaMemcpy(d_packedBwt, idx->packedBwt, packedBwtSize, cudaMemcpyHostToDevice));
    size += packedBwtSize;

    hostFmIndex.oneHot = d_one_hot;
    hostFmIndex.cpOcc = d_cp_occ;
    hostFmIndex.cpOcc2 = d_cp_occ2;
    hostFmIndex.count = d_count;
    hostFmIndex.count2 = d_count2;
    hostFmIndex.firstBase = d_first_base;
    hostFmIndex.sentinelIndex = deviceSentinelIndex;
    hostFmIndex.suffixArrayMsByte = d_suffixArrayMsByte;
    hostFmIndex.suffixArrayLsWord = d_suffixArrayLsWord;
    hostFmIndex.referenceLen = d_referenceLen;
    hostFmIndex.packedBwt = d_packedBwt;

    fmindex_t *deviceFmIndex;
    CUDA_CHECK(cudaMalloc((void**)&deviceFmIndex, sizeof(fmindex_t)));
    CUDA_CHECK(cudaMemcpy(deviceFmIndex, &hostFmIndex, sizeof(fmindex_t), cudaMemcpyHostToDevice));

    std::cerr << "* occ2 index: " << size / MB_SIZE << " MB\n";

    // output
    process_instance->d_fmi = deviceFmIndex;
}




process_data_t * device_alloc(
        int gpuid,
        pipeline_aux_t *aux
        )
{
    CUDA_CHECK(cudaSetDevice(gpuid));
    int current;
    CUDA_CHECK(cudaGetDevice(&current));
    if(current != gpuid){
        exit(1);
    }
    process_data_t *proc = new process_data_t;
    proc->gpu_no = gpuid;
    size_t size = 0;  // tracks total device VRAM for intermediates (excludes pool and index)

    // Right-size per-read arrays to the actual -Z batch size rather than the compile-time cap.
    // MAX_BATCH_SIZE is the hard ceiling enforced by memcpy_input; g3_opt->batch_size is the
    // user-requested value (-Z flag) which is always <= MAX_BATCH_SIZE.
    int max_reads = (aux->g3_opt->batch_size > 0 && aux->g3_opt->batch_size <= MAX_BATCH_SIZE)
                   ? aux->g3_opt->batch_size : MAX_BATCH_SIZE;

	// dynamic allocation pool management
    long long pool_cap = aux->g3_opt->pool_mb * (long long)MB_SIZE;
	proc->d_buffer_pools = CUDA_BufferInit(pool_cap);
    std::cerr << "* device " << gpuid << " allocating "
        << pool_cap / MB_SIZE << " MB for dynamic allocation pool\n";

    size_t seed_recs_bytes = (size_t)max_reads * MAX_NUM_SW_SEEDS * sizeof(seed_record_t);
    CUDA_CHECK(cudaMalloc(&proc->d_aux,          sizeof(smem_aux_t) * max_reads));  size += sizeof(smem_aux_t) * max_reads;
    CUDA_CHECK(cudaMalloc(&proc->d_seq_seeds,    sizeof(mem_seed_v) * max_reads));  size += sizeof(mem_seed_v) * max_reads;
    CUDA_CHECK(cudaMalloc(&proc->d_seed_records, seed_recs_bytes));                  size += seed_recs_bytes;
    proc->d_lifetime_arena = nullptr;

    // Per-read arrays right-sized to max_reads
	CUDA_CHECK(cudaMalloc(&proc->d_chains, max_reads * sizeof(mem_chain_v)));   size += max_reads * sizeof(mem_chain_v);
	// Use host-mapped (zero-copy) memory for d_Nseeds and d_Nalns so kernels
	// write directly to host-visible memory, eliminating D2H cudaMemcpy stalls.
	CUDA_CHECK(cudaHostAlloc((void**)&proc->h_Nseeds, sizeof(int), cudaHostAllocMapped));
	CUDA_CHECK(cudaHostGetDevicePointer((void**)&proc->d_Nseeds, proc->h_Nseeds, 0));
	CUDA_CHECK(cudaHostAlloc((void**)&proc->h_Nalns, sizeof(int), cudaHostAllocMapped));
	CUDA_CHECK(cudaHostGetDevicePointer((void**)&proc->d_Nalns, proc->h_Nalns, 0));
    // +1: compute_offsets_on_host (myhelper.cuh) runs exclusive_scan over [regs, regs+batch_size+1)
    // which reads d_regs[batch_size] as a sentinel and writes d_offsets[batch_size] as the total.
    // Allocating max_reads+1 ensures the sentinel index is within bounds when batch_size==max_reads.
	CUDA_CHECK(cudaMalloc(&proc->d_regs,    (max_reads + 1) * sizeof(mem_alnreg_v)));  size += (max_reads + 1) * sizeof(mem_alnreg_v);
	CUDA_CHECK(cudaMalloc(&proc->d_alns,     max_reads      * sizeof(mem_aln_v)));      size += max_reads * sizeof(mem_aln_v);
	CUDA_CHECK(cudaMalloc(&proc->d_offsets, (max_reads + 1) * sizeof(int)));            size += (max_reads + 1) * sizeof(int);

	// Output arrays indexed by alignment ID (not read ID): total alignments per batch can
	// exceed max_reads for hg38 (many secondary alignments per read). Use MAX_ALN_CNT.
	CUDA_CHECK(cudaMalloc(&proc->d_rids,      MAX_ALN_CNT * sizeof(int)));                   size += (size_t)MAX_ALN_CNT * sizeof(int);
	CUDA_CHECK(cudaMalloc(&proc->d_positions, MAX_ALN_CNT * sizeof(int64_t)));               size += (size_t)MAX_ALN_CNT * sizeof(int64_t);
	CUDA_CHECK(cudaMalloc(&proc->d_ncigars,   MAX_ALN_CNT * sizeof(int)));                   size += (size_t)MAX_ALN_CNT * sizeof(int);
	CUDA_CHECK(cudaMalloc(&proc->d_cigars,    MAX_ALN_CNT * sizeof(uint32_t) * MAX_N_CIGAR)); size += (size_t)MAX_ALN_CNT * sizeof(uint32_t) * MAX_N_CIGAR;
	CUDA_CHECK(cudaMalloc(&proc->d_flags,     MAX_ALN_CNT * sizeof(int)));                   size += (size_t)MAX_ALN_CNT * sizeof(int);
	CUDA_CHECK(cudaMalloc(&proc->d_mapqs,     MAX_ALN_CNT * sizeof(int)));                   size += (size_t)MAX_ALN_CNT * sizeof(int);

	// Sorting arrays indexed by alignment ID; must match MAX_ALN_CNT.
	CUDA_CHECK(cudaMalloc(&proc->d_sortkeys_in,  MAX_ALN_CNT * sizeof(int)));  size += (size_t)MAX_ALN_CNT * sizeof(int);
	CUDA_CHECK(cudaMalloc(&proc->d_sortkeys_out, MAX_ALN_CNT * sizeof(int)));  size += (size_t)MAX_ALN_CNT * sizeof(int);
	CUDA_CHECK(cudaMalloc(&proc->d_seqIDs_in,    MAX_ALN_CNT * sizeof(int)));  size += (size_t)MAX_ALN_CNT * sizeof(int);
	CUDA_CHECK(cudaMalloc(&proc->d_seqIDs_out,   MAX_ALN_CNT * sizeof(int)));  size += (size_t)MAX_ALN_CNT * sizeof(int);

    if (proc->d_aux && proc->d_seq_seeds && proc->d_seed_records && proc->d_chains
            && proc->d_Nseeds && proc->d_Nalns && proc->d_regs
            && proc->d_offsets && proc->d_rids && proc->d_positions && proc->d_ncigars && proc->d_cigars
            && proc->d_alns && proc->d_sortkeys_in && proc->d_sortkeys_out
            && proc->d_seqIDs_in && proc->d_seqIDs_out) {
        std::cerr << "* device " << gpuid
            << " intermediate data " << size / MB_SIZE << " MB\n";
    } else {
        std::cerr << "* device " << gpuid
            << " intermediate data alloc failed\n";
        exit(EXIT_FAILURE);
    }

    // input on device (sized at MAX_BATCH_SIZE — fixed input ring buffer, independent of -Z)
    {
        size_t ssize = (size_t)MAX_BATCH_SIZE * sizeof(uint8_t) * MAX_LEN_READ;
        CUDA_CHECK(cudaMalloc(&proc->d_seq, ssize));
        size += ssize;
        std::cerr << "* device " << gpuid << " allocating "
            << ssize / MB_SIZE << " MB for input seqs\n";
        if (!proc->d_seq) { std::cerr << "Error.  device memory for copying seqs.\n"; exit(EXIT_FAILURE); }
    }
    {
        size_t ssize = sizeof(int) * (MAX_BATCH_SIZE + 1);
        CUDA_CHECK(cudaMalloc(&proc->d_seq_offset, ssize));
        size += ssize;
        std::cerr << "* device " << gpuid << " allocating "
            << ssize / MB_SIZE << " MB for input seq offsets\n";
        if (!proc->d_seq_offset) { std::cerr << "Error.  device memory for copying offsets.\n"; exit(EXIT_FAILURE); }
    }

    CUDA_CHECK(cudaMalloc(&proc->d_alns_offset,     sizeof(int)      * (max_reads + 1)));       size += sizeof(int) * (max_reads + 1);
    CUDA_CHECK(cudaMalloc(&proc->d_rid,             sizeof(int)      * (MAX_ALN_CNT + 1)));      size += sizeof(int) * (size_t)(MAX_ALN_CNT + 1);
    CUDA_CHECK(cudaMalloc(&proc->d_pos,             sizeof(uint64_t) * (MAX_ALN_CNT + 1)));      size += sizeof(uint64_t) * (size_t)(MAX_ALN_CNT + 1);
    CUDA_CHECK(cudaMalloc(&proc->d_chunk_aln_count, sizeof(int)));                               size += sizeof(int);
    if (!proc->d_alns_offset || !proc->d_rid || !proc->d_pos || !proc->d_chunk_aln_count) {
        std::cerr << "cudaMalloc err.\n"; exit(EXIT_FAILURE);
    }

    proc->intermediate_bytes = size;

	// pinned memory for async H2D memcpy of input sequences (cudaHostAlloc — no device VRAM)
    {
        size_t ssize = (size_t)MAX_BATCH_SIZE * sizeof(uint8_t) * MAX_LEN_READ;
        CUDA_CHECK(cudaHostAlloc(&proc->h_seq, ssize, cudaHostAllocDefault));
        std::cerr << "* allocating " << ssize / MB_SIZE << " MB for input seq pinned memcpy\n";
        if (!proc->h_seq) { std::cerr << "Error. Host pinned memory for copying seqs.\n"; exit(EXIT_FAILURE); }
    }
    {
        size_t ssize = (size_t)(MAX_BATCH_SIZE + 1) * sizeof(int);
        CUDA_CHECK(cudaHostAlloc(&proc->h_seq_offset, ssize, cudaHostAllocDefault));
        std::cerr << "* allocating " << ssize / MB_SIZE << " MB for input seq offsets pinned memcpy\n";
        if (!proc->h_seq_offset) { std::cerr << "Error. Host pinned memory for copying offsets.\n"; exit(EXIT_FAILURE); }
    }

    // pinned host output buffers for truly async D2H transfers.
    // d_offsets is sync-copied first (need count before async copies can be sized).
    CUDA_CHECK(cudaHostAlloc(&proc->h_out_offsets,   (size_t)(MAX_BATCH_SIZE + 1) * sizeof(int),      cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&proc->h_out_rids,      (size_t)MAX_ALN_CNT * sizeof(int),               cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&proc->h_out_positions, (size_t)MAX_ALN_CNT * sizeof(int64_t),           cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&proc->h_out_ncigars,   (size_t)MAX_ALN_CNT * sizeof(int),               cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&proc->h_out_cigars,    (size_t)MAX_ALN_CNT * MAX_N_CIGAR * sizeof(uint32_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&proc->h_out_flags,     (size_t)MAX_ALN_CNT * sizeof(int),               cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&proc->h_out_mapqs,     (size_t)MAX_ALN_CNT * sizeof(int),               cudaHostAllocDefault));

	// initialize a cuda stream for processing
	proc->CUDA_stream = malloc(sizeof(cudaStream_t));
	CUDA_CHECK(cudaStreamCreate((cudaStream_t*)proc->CUDA_stream));

    // Preallocate CUB temp storage for DeviceRadixSort (used per-batch in bwamem.cu).
    // Query required size once using MAX_ALN_CNT as the upper bound.
    {
        int    *dummy_keys_in  = nullptr, *dummy_keys_out  = nullptr;
        int    *dummy_vals_in  = nullptr, *dummy_vals_out  = nullptr;
        proc->d_cub_temp       = nullptr;
        proc->d_cub_temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            proc->d_cub_temp, proc->d_cub_temp_bytes,
            dummy_keys_in, dummy_keys_out,
            dummy_vals_in, dummy_vals_out,
            MAX_ALN_CNT, 0, 8 * sizeof(int));
        CUDA_CHECK(cudaMalloc(&proc->d_cub_temp, proc->d_cub_temp_bytes));
    }

    // Preallocate GSS (gather-sort-scatter) buffers for sa_lookup_kernel.
    // Upper-bound: max_reads × SAL_SCAN_CAPACITY (512 intervals × max_occ=500 seeds each).
    // In practice hg38 batches see ~20 seeds/read on average; 512×500 is a hard ceiling.
    // If a batch exceeds the provisioned bound, GSS falls back to the original path for that batch.
    {
        // 100 seeds/read: ~90th-percentile for hg38 148bp batches (max_occ=500 but typical much lower).
        // If a batch exceeds this, GSS falls back gracefully to the original path.
        const int SAL_GSS_SEEDS_PER_READ = 100;
        int gss_max = max_reads * SAL_GSS_SEEDS_PER_READ;
        proc->gss_max_seeds = gss_max;

        CUDA_CHECK(cudaMalloc(&proc->d_per_read_total_seeds, (max_reads + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&proc->d_read_base_offsets,   (max_reads + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&proc->d_sal_keys,            gss_max * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&proc->d_sal_keys_sorted,     gss_max * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&proc->d_sal_vals,            gss_max * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&proc->d_sal_perm,            gss_max * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&proc->d_sal_meta_buf,        2 * gss_max * sizeof(sal_meta_t)));
        CUDA_CHECK(cudaMalloc(&proc->d_sal_rbeg,            gss_max * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&proc->d_seq_actual_count,    max_reads * sizeof(int)));

        // CUB temp for SortPairs(uint64 keys × int vals, gss_max elements).
        proc->d_gss_cub_temp       = nullptr;
        proc->d_gss_cub_temp_bytes = 0;
        {
            uint64_t *dummy_k_in  = nullptr, *dummy_k_out = nullptr;
            int      *dummy_v_in  = nullptr, *dummy_v_out = nullptr;
            cub::DeviceRadixSort::SortPairs(
                proc->d_gss_cub_temp, proc->d_gss_cub_temp_bytes,
                dummy_k_in, dummy_k_out,
                dummy_v_in, dummy_v_out,
                gss_max, 0, sizeof(uint64_t) * 8);
        }
        // Also need temp for DeviceScan::ExclusiveSum(int, max_reads+1).
        size_t scan_tmp_bytes = 0;
        {
            int *dummy_in = nullptr, *dummy_out = nullptr;
            cub::DeviceScan::ExclusiveSum(nullptr, scan_tmp_bytes,
                dummy_in, dummy_out, max_reads + 1);
        }
        size_t gss_cub_bytes = std::max(proc->d_gss_cub_temp_bytes, scan_tmp_bytes);
        proc->d_gss_cub_temp_bytes = gss_cub_bytes;
        CUDA_CHECK(cudaMalloc(&proc->d_gss_cub_temp, gss_cub_bytes));

        size_t gss_bytes = (size_t)(max_reads + 1) * sizeof(int) * 2   // per_read + base_offsets
            + (size_t)gss_max * (sizeof(uint64_t)*2 + sizeof(int)*2 + sizeof(sal_meta_t)*2 + sizeof(int64_t))
            + (size_t)max_reads * sizeof(int)  // actual_count
            + gss_cub_bytes;
        std::cerr << "* device " << gpuid << " GSS buffers "
                  << gss_bytes / MB_SIZE << " MB (max " << gss_max << " seeds)\n";
    }

    return proc;
}

void memcpy_index(
        process_data_t *instance,
        int gpuid, 
        pipeline_aux_t *aux
        )
{
    TIMER_INIT();
    TIMER_START();
    CUDA_CHECK(cudaSetDevice(gpuid));
    int current;
    CUDA_CHECK(cudaGetDevice(&current));
    if(current != gpuid){
        std::cerr << "GPU " << gpuid << "  " << "* device_alloc: cudaSetDevice is wrong" << std::endl;
        exit(1);
    }
    unsigned long long size;

	// user-defined options
	transferOptions(aux->opt, aux->pes0, instance, &size);
    
	// transfer index data
	transferIndex(aux->idx->bns, aux->idx->pac,
            aux->kmerHashTab, instance, &size);

    transferFmIndex(instance, &(aux->loadedIndex), &size);

    // decide at index-load time whether to use GSS for sa_lookup_kernel.
    // GSS improves L2 hit rate by sorting BWT positions, but adds CUB sort overhead.
    // Only worth it when the SA table >> L2 cache capacity (hg38: SA ~6 GB >> L2;
    // E. coli: SA ~12 MB ≈ L2 → original path faster).
    // Threshold: SA table > 100 MB (SA_COMPX=8 compressed, 5B/entry: refLen/8 × 5B).
    {
        int64_t refLen = *(aux->loadedIndex.referenceLen);
        int64_t sa_num_entries = (refLen >> SA_COMPX) + 1;
        size_t  sa_bytes = sa_num_entries * (sizeof(int8_t) + sizeof(uint32_t));  // MsByte + LsWord
        const size_t SAL_GSS_SA_THRESHOLD = (size_t)100 * 1024 * 1024;  // 100 MB
        instance->use_gss = (sa_bytes > SAL_GSS_SA_THRESHOLD);
        std::cerr << "* SA table " << sa_bytes / MB_SIZE << " MB → GSS="
                  << (instance->use_gss ? "ON" : "OFF") << "\n";
    }

    TIMER_END(0, "");
    tprof[gpuid][GPU_SETUP] = duration.count() / 1000;
}


void memcpy_input(int batch_size, process_data_t *proc,
        uint8_t *seq, int *seq_offset)
{
    cudaStream_t stream = *(cudaStream_t*)proc->CUDA_stream;
    int total_seq_len = seq_offset[batch_size];
    size_t size;

    cudaGetLastError();  // clear any sticky error from index loading before starting batch copies

    // Copy into pinned staging buffers then transfer async on the compute stream
    size = sizeof(uint8_t) * total_seq_len;
    memcpy(proc->h_seq, seq, size);
    CUDA_CHECK(cudaMemcpyAsync(proc->d_seq, proc->h_seq, size,
            cudaMemcpyHostToDevice, stream));

    if (batch_size > MAX_BATCH_SIZE) {
        std::cerr << "[ERROR] batch_size=" << batch_size << " > MAX_BATCH_SIZE=" << MAX_BATCH_SIZE
                  << " — reduce -F (bytes-per-record estimate) or increase MAX_BATCH_SIZE in macro.h\n";
        exit(EXIT_FAILURE);
    }
    size = sizeof(int) * (batch_size + 1);
    memcpy(proc->h_seq_offset, seq_offset, size);
    CUDA_CHECK(cudaMemcpyAsync(proc->d_seq_offset, proc->h_seq_offset, size,
            cudaMemcpyHostToDevice, stream));

    proc->batch_size = batch_size;

    // reset — d_Nseeds is host-mapped, so write directly through the host pointer
    *proc->h_Nseeds = 0;
    // reset pool headers asynchronously on the compute stream
    CUDAResetBufferPoolAsync(proc->d_buffer_pools, stream);
}




void check_device_count(int num_requested_gpus)
{
    int num_available_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_available_gpus));
    if(num_available_gpus < num_requested_gpus){
        std::cerr << "!! invalid request of " << num_requested_gpus 
            << " GPUs where only " << num_available_gpus << " GPUs are available.";
        exit(1);
    } else{
        std::cerr << "* using " << num_requested_gpus << 
            " GPUs out of " << num_available_gpus << " available GPUs.\n";
    }
}

void destruct_proc(process_data_t *proc)
{
    if(proc->CUDA_stream){
        cudaStreamDestroy(*(cudaStream_t*)proc->CUDA_stream);
        free(proc->CUDA_stream);
        proc->CUDA_stream = nullptr;
    }
    if(proc->h_Nseeds) cudaFreeHost(proc->h_Nseeds);
    if(proc->h_Nalns)  cudaFreeHost(proc->h_Nalns);
    if(proc->h_seq)        cudaFreeHost(proc->h_seq);
    if(proc->h_seq_offset) cudaFreeHost(proc->h_seq_offset);
    if(proc->h_out_offsets)   cudaFreeHost(proc->h_out_offsets);
    if(proc->h_out_rids)      cudaFreeHost(proc->h_out_rids);
    if(proc->h_out_positions) cudaFreeHost(proc->h_out_positions);
    if(proc->h_out_ncigars)   cudaFreeHost(proc->h_out_ncigars);
    if(proc->h_out_cigars)    cudaFreeHost(proc->h_out_cigars);
    if(proc->h_out_flags)     cudaFreeHost(proc->h_out_flags);
    if(proc->h_out_mapqs)     cudaFreeHost(proc->h_out_mapqs);
    if(proc->d_cub_temp)   cudaFree(proc->d_cub_temp);
    // GSS buffers
    if(proc->d_per_read_total_seeds) cudaFree(proc->d_per_read_total_seeds);
    if(proc->d_read_base_offsets)    cudaFree(proc->d_read_base_offsets);
    if(proc->d_sal_keys)             cudaFree(proc->d_sal_keys);
    if(proc->d_sal_keys_sorted)      cudaFree(proc->d_sal_keys_sorted);
    if(proc->d_sal_vals)             cudaFree(proc->d_sal_vals);
    if(proc->d_sal_perm)             cudaFree(proc->d_sal_perm);
    if(proc->d_sal_meta_buf)         cudaFree(proc->d_sal_meta_buf);
    if(proc->d_sal_rbeg)             cudaFree(proc->d_sal_rbeg);
    if(proc->d_seq_actual_count)     cudaFree(proc->d_seq_actual_count);
    if(proc->d_gss_cub_temp)         cudaFree(proc->d_gss_cub_temp);
}

size_t get_free_vram(void)
{
    size_t free_vram, total_vram;
    cudaMemGetInfo(&free_vram, &total_vram);
    return free_vram;
}

/* share_index — copy read-only index device pointers from src to dst.
 *
 * Called for the second (ping-pong) buffer after memcpy_index loads the
 * index into buffer 0.  Both buffers share the same device memory for
 * d_bns, d_pac, d_kmerHashTab, d_opt, d_pes, and d_fmi — no duplication.
 * destruct_proc does not free these pointers so there is no double-free risk.
 */
void share_index(process_data_t *dst, const process_data_t *src)
{
    if (dst == src) return; // same buffer (single-buffer fallback), nothing to do
    dst->d_opt         = src->d_opt;
    dst->d_pes         = src->d_pes;
    dst->d_bns         = src->d_bns;
    dst->d_pac         = src->d_pac;
    dst->d_kmerHashTab = src->d_kmerHashTab;
    dst->d_fmi         = src->d_fmi;
    dst->use_gss       = src->use_gss;  // propagate GSS flag
}

void cuda_wrapper_test()
{
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<long long, std::micro> duration;
    start = std::chrono::high_resolution_clock::now();

    cudaFree(0);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cerr << "* cuda init: " << duration.count() / 1000 << " ms" << std::endl;
}


void memcpy_output(
        aligned_chunk_t *ac,
        process_data_t * proc)
{
    // copy output from device to host
    // proc->d_offsets, proc->d_rids, proc->d_positions, proc->d_ncigars, proc->d_cigars.
    // ac->offsets, ac->rids, ac->positions, ac->ncigars, ac->cigars
    ac->offsets.resize(ac->chunk_size + 1); // + 1 for the total count
    CUDA_CHECK(cudaMemcpy(ac->offsets.data(),
                proc->d_offsets,
                sizeof(int) * (ac->chunk_size + 1),
                cudaMemcpyDeviceToHost));
    int batch_alns_count = ac->offsets[ac->chunk_size];
    ac->rids.resize(batch_alns_count);
    ac->positions.resize(batch_alns_count);
    ac->ncigars.resize(batch_alns_count);
    ac->cigars.resize(batch_alns_count * MAX_N_CIGAR);
    ac->flags.resize(batch_alns_count);
    ac->mapqs.resize(batch_alns_count);

    CUDA_CHECK(cudaMemcpy(ac->rids.data(),
                proc->d_rids,
                sizeof(int) * batch_alns_count,
                cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ac->positions.data(),
                proc->d_positions,
                sizeof(int64_t) * batch_alns_count,
                cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ac->ncigars.data(),
                proc->d_ncigars,
                sizeof(int) * batch_alns_count,
                cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ac->cigars.data(),
                proc->d_cigars,
                sizeof(uint32_t) * batch_alns_count * MAX_N_CIGAR,
                cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ac->flags.data(),
                proc->d_flags,
                sizeof(int) * batch_alns_count,
                cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ac->mapqs.data(),
                proc->d_mapqs,
                sizeof(int) * batch_alns_count,
                cudaMemcpyDeviceToHost));
}


// async D2H to per-slot pinned host arrays.
// o_stream_opaque: dedicated offsets-only stream (kept empty between calls; no prior ops).
//   cudaMemcpyAsync on o_stream + cudaStreamSynchronize is stream-specific (no device sync),
//   so it waits only for the ~195 KB offsets copy, not d2h_stream's bulk ops.
// d2h_stream_opaque: bulk data stream (overlaps with next batch's GPU compute).
// Caller must cudaEventRecord(d2h_done, d2h_stream) immediately after this call, then
// cudaEventSynchronize(d2h_done) in worker_samgen before reading the arrays.
void memcpy_output_async(
        aligned_chunk_t *ac,
        process_data_t  *proc,
        void            *o_stream_opaque,
        void            *d2h_stream_opaque)
{
    cudaStream_t o_stream   = *(cudaStream_t*)o_stream_opaque;
    cudaStream_t d2h_stream = *(cudaStream_t*)d2h_stream_opaque;

    // 1. Async-copy d_offsets on o_stream (no prior ops → stream-specific sync is fast).
    size_t offsets_bytes = sizeof(int) * (ac->chunk_size + 1);
    CUDA_CHECK(cudaMemcpyAsync(proc->h_out_offsets,
                proc->d_offsets,
                offsets_bytes,
                cudaMemcpyDeviceToHost, o_stream));
    CUDA_CHECK(cudaStreamSynchronize(o_stream)); // stream-specific: waits only for o_stream
    // Copy to ac->offsets for worker_samgen (reads ac->offsets for seq_id loop).
    ac->offsets.resize(ac->chunk_size + 1);
    memcpy(ac->offsets.data(), proc->h_out_offsets, offsets_bytes);
    int batch_alns_count = proc->h_out_offsets[ac->chunk_size];

    // 2. Async-copy remaining output to per-slot pinned arrays (truly non-blocking DMA).
    //    worker_samgen reads from proc->h_out_* after cudaEventSynchronize(d2h_done).
    CUDA_CHECK(cudaMemcpyAsync(proc->h_out_rids,
                proc->d_rids,
                sizeof(int) * batch_alns_count,
                cudaMemcpyDeviceToHost, d2h_stream));
    CUDA_CHECK(cudaMemcpyAsync(proc->h_out_positions,
                proc->d_positions,
                sizeof(int64_t) * batch_alns_count,
                cudaMemcpyDeviceToHost, d2h_stream));
    CUDA_CHECK(cudaMemcpyAsync(proc->h_out_ncigars,
                proc->d_ncigars,
                sizeof(int) * batch_alns_count,
                cudaMemcpyDeviceToHost, d2h_stream));
    CUDA_CHECK(cudaMemcpyAsync(proc->h_out_cigars,
                proc->d_cigars,
                sizeof(uint32_t) * batch_alns_count * MAX_N_CIGAR,
                cudaMemcpyDeviceToHost, d2h_stream));
    CUDA_CHECK(cudaMemcpyAsync(proc->h_out_flags,
                proc->d_flags,
                sizeof(int) * batch_alns_count,
                cudaMemcpyDeviceToHost, d2h_stream));
    CUDA_CHECK(cudaMemcpyAsync(proc->h_out_mapqs,
                proc->d_mapqs,
                sizeof(int) * batch_alns_count,
                cudaMemcpyDeviceToHost, d2h_stream));
}

// opaque CUDA stream/event wrappers — callable from g++-compiled pipeline.cpp.
void *cuda_d2h_stream_create(int gpuid) {
    CUDA_CHECK(cudaSetDevice(gpuid));
    cudaStream_t *s = (cudaStream_t*)malloc(sizeof(cudaStream_t));
    CUDA_CHECK(cudaStreamCreate(s));
    return s;
}

void cuda_d2h_stream_destroy(void *s) {
    CUDA_CHECK(cudaStreamDestroy(*(cudaStream_t*)s));
    free(s);
}

void *cuda_o_stream_create(int gpuid) {
    CUDA_CHECK(cudaSetDevice(gpuid));
    cudaStream_t *s = (cudaStream_t*)malloc(sizeof(cudaStream_t));
    CUDA_CHECK(cudaStreamCreate(s));
    return s;
}

void cuda_o_stream_destroy(void *s) {
    CUDA_CHECK(cudaStreamDestroy(*(cudaStream_t*)s));
    free(s);
}

void *cuda_d2h_event_create(void) {
    cudaEvent_t *e = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
    CUDA_CHECK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
    return e;
}

void cuda_d2h_event_destroy(void *e) {
    CUDA_CHECK(cudaEventDestroy(*(cudaEvent_t*)e));
    free(e);
}

void cuda_d2h_event_record(void *e, void *s) {
    CUDA_CHECK(cudaEventRecord(*(cudaEvent_t*)e, *(cudaStream_t*)s));
}

void cuda_d2h_event_sync(void *e) {
    CUDA_CHECK(cudaEventSynchronize(*(cudaEvent_t*)e));
}

void cuda_set_device(int gpuid) {
    CUDA_CHECK(cudaSetDevice(gpuid));
}
