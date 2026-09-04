#include "cuda_wrapper.h"
#include "gmem_alloc.cuh"

#include "errchk.cuh"
#include "utils_debug.cuh"
#include "macro.h"
#include "timer.h"
#include "myhelper.cuh"

#include "gpu_types.h"
#include "bntseq.h"
#include "fastmap.h"

#include "hash_kmer_index.h"
#include "seed.cuh"
#include "chain.cuh"
#include "extend.cuh"
#include "region.cuh"
#include "finalize.cuh"

#include <string.h>
#include <fstream>
#include <iostream>
#include <cstdlib>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define PRINT(LABEL) \
	g3_opt->print_mask & BIT(LABEL)

// Kernel timing: gated by G3_KERNEL_TIMING=1 env var (default off).
// Without gating, CUDA_TIMER_END calls cudaEventSynchronize after every kernel, inserting
// one host-blocking sync per timed region (15 per batch × N batches = O(5000) stalls on hg38).
// When disabled, tprof[gpuid][...] += 0 on every batch — no behavior change, timer stays 0.
static bool g_kernel_timing = (getenv("G3_KERNEL_TIMING") != nullptr);

#define CUDA_TIMER_INIT \
	cudaEvent_t timer_event_start = nullptr, timer_event_stop = nullptr;\
	if (g_kernel_timing) {\
	    CUDA_CHECK(cudaEventCreate(&timer_event_start));\
	    CUDA_CHECK(cudaEventCreate(&timer_event_stop));\
	}

#define CUDA_TIMER_DESTROY \
	if (g_kernel_timing) {\
	    CUDA_CHECK(cudaEventDestroy(timer_event_start));\
	    CUDA_CHECK(cudaEventDestroy(timer_event_stop));\
	}

#define CUDA_TIMER_START(lap) \
	lap = 0;\
	if (g_kernel_timing) { CUDA_CHECK(cudaEventRecord(timer_event_start, stream)); }

#define CUDA_TIMER_END(lap) \
	if (g_kernel_timing) {\
	    CUDA_CHECK(cudaEventRecord(timer_event_stop, stream));\
	    CUDA_CHECK(cudaEventSynchronize(timer_event_stop));\
	    CUDA_CHECK(cudaEventElapsedTime(&lap, timer_event_start, timer_event_stop));\
	}

#define CUDA_CHECK_KERNEL_RUN()\
{\
	cudaError_t err;\
	err = cudaGetLastError();\
	if(err != cudaSuccess)\
	{\
		fprintf(stderr,"GPU %d cudaGetLastError(): %s %s %d\n", gpuid, cudaGetErrorString(err), __FILE__, __LINE__);\
		return 1;\
	}\
}


extern float tprof[MAX_NUM_GPUS][MAX_NUM_STEPS];

/*
 *
 * Stage    |   Substage        |   Step          |   note
 * ---------------------------------------------------------------------------
 * Seeding  |   SMEM seeding    |   seed          |   seeding core
 *          |   Reseeding       |   r2            |
 *          |                   |   r3            |
 * Chaining |   B-tree chaining |   sal           |
 *          |                   |   sort_seeds    |
 *          |                   |   chain         |   chaining core
 *          |                   |   sort_chains   |
 *          |                   |   filter        |
 * Extending|   Local extending |   pairgen       |
 *          |                   |   extend        |   extending core
 *          |                   |   filter_mark   |
 *          |                   |   sort_alns     |
 *          |   Traceback       |   pairgen       |
 *          |                   |   traceback     |   traceback core (w/ NM test)
 *          |                   |   finalize      |
 */
int bwa_align(int gpuid, process_data_t *proc, g3_opt_t *g3_opt)
{
	float step_lap;
	void *d_temp_storage;
	size_t temp_storage_size;
	int batch_size, num_seeds_to_extend;
	int batch_num_alns;
	if((batch_size = proc->batch_size) == 0){
		return 0;
	}
	cudaStream_t stream = *(cudaStream_t*)proc->CUDA_stream;
	CUDA_TIMER_INIT;
	// (1/3) Seeding
	// SMEM seeding (Seeding 1/2)
	CUDA_TIMER_START(step_lap);
	preseed_and_filter <<< batch_size, REGION_FILTER_BLOCKSIZE, 0, stream >>> (
			proc->d_fmi,
			proc->d_opt, proc->d_seq, proc->d_seq_offset,
			proc->d_aux, proc->d_kmerHashTab, proc->d_buffer_pools);
	CUDA_CHECK_KERNEL_RUN();
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][S_SMEM] += step_lap;
	if(PRINT(_SMEM) || PRINT(_ALL_SEEDING)){
		for(int readID = 0; readID < batch_size; readID++)
		{
			printIntv<<<1, WARPSIZE, 0, stream>>>(proc->d_aux, readID, _SMEM);
			CUDA_CHECK_KERNEL_RUN();
		}
	}
	// Reseeding (Seeding 2/2): 1-warp/block, <<<batch_size, 32>>>.
	CUDA_TIMER_START(step_lap);
	reseedV2 <<< batch_size, WARPSIZE, 0, stream >>>(
				proc->d_fmi, proc->d_opt, proc->d_seq, proc->d_seq_offset,
				proc->d_aux, proc->d_kmerHashTab,
				proc->d_buffer_pools, batch_size);
	CUDA_CHECK_KERNEL_RUN();
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][S_R2] += step_lap;
	// Reseeding 3rd Round
	CUDA_TIMER_START(step_lap);
	reseedLastRound <<< batch_size, WARPSIZE, 0, stream >>>(
				proc->d_fmi, proc->d_opt, proc->d_seq,
			       	proc->d_seq_offset,
				proc->d_aux, proc->d_kmerHashTab, batch_size);
	CUDA_CHECK_KERNEL_RUN();
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][S_R3] += step_lap;
	if(PRINT(_INTV) || PRINT(_ALL_SEEDING)){
		for(int readID = 0; readID < batch_size; readID++)
		{
			printIntv<<<1, WARPSIZE, 0, stream>>>(proc->d_aux, readID, _INTV);
			CUDA_CHECK_KERNEL_RUN();
		}
	}
	// (2/3) Chaining
	// B-tree chaining (Chaining 1/1)
	// Gather-Sort-Scatter path for sa_lookup_kernel.
	// Runs when proc->use_gss is true (SA table >> L2 cache, set at index-load time)
	// AND average read length >= SAL_GSS_MIN_READ_LEN.
	// For short reads (76 bp), the GSS prepass overhead (two kernels + CUB scan +
	// cudaStreamSynchronize + D2H copy) exceeds the savings; skip directly to the
	// original per-read path.  For long reads (148 bp+), GSS is a net win.
	// avg_read_len is computed from h_seq_offset[], which is already populated by
	// memcpy_input() before bwa_align() is called — no extra host-device transfer needed.
	CUDA_TIMER_START(step_lap);
	{
		int total_seq_len = (batch_size > 0)
				? (proc->h_seq_offset[batch_size] - proc->h_seq_offset[0])
				: 0;
		int avg_read_len = (batch_size > 0) ? (total_seq_len / batch_size) : 0;
		// G3_FORCE_GSS bypasses all gates (including read-length) for testing.
		bool maybe_use_gss = getenv("G3_FORCE_GSS")
				|| (proc->use_gss && avg_read_len >= SAL_GSS_MIN_READ_LEN);
		if (getenv("G3_GSS_DEBUG")) fprintf(stderr, "[GSS] batch_size=%d avg_read_len=%d use_gss=%d maybe_use_gss=%d\n", batch_size, avg_read_len, (int)proc->use_gss, (int)maybe_use_gss);

		if (maybe_use_gss) {
		// Step 1: count total seeds per read → d_per_read_total_seeds[].
		sa_lookup_prepass_count_kernel <<< batch_size, SAL_SORT_BLOCKDIMX, 0, stream >>> (
				proc->d_opt, proc->d_aux, proc->d_per_read_total_seeds);
		CUDA_CHECK_KERNEL_RUN();

		// Step 2: exclusive prefix sum → d_read_base_offsets[0..batch_size].
		//   d_read_base_offsets[batch_size] = batch_total_seeds.
		// Zero the padding element so ExclusiveSum(batch_size+1) reads it as 0.
		CUDA_CHECK(cudaMemsetAsync(proc->d_per_read_total_seeds + batch_size,
				0, sizeof(int), stream));
		{
			void  *d_tmp   = proc->d_gss_cub_temp;
			size_t tmp_bytes = proc->d_gss_cub_temp_bytes;
			CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
					d_tmp, tmp_bytes,
					proc->d_per_read_total_seeds,
					proc->d_read_base_offsets,
					batch_size + 1, stream));
		}
		CUDA_CHECK_KERNEL_RUN();

		// Step 3: read batch_total_seeds (1 int D2H) to decide which path to take.
		int batch_total_seeds = 0;
		CUDA_CHECK(cudaStreamSynchronize(stream));
		CUDA_CHECK(cudaMemcpy(&batch_total_seeds,
				proc->d_read_base_offsets + batch_size,
				sizeof(int), cudaMemcpyDeviceToHost));

		bool do_gss = batch_total_seeds > 0 && batch_total_seeds <= proc->gss_max_seeds;
		if (getenv("G3_GSS_DEBUG")) fprintf(stderr, "[GSS] batch_size=%d batch_total_seeds=%d use_gss=%d max=%d path=%s\n", batch_size, batch_total_seeds, (int)proc->use_gss, proc->gss_max_seeds, do_gss ? "GSS" : "orig");
		if (do_gss) {
			// === GSS path ===
			// Step 4: emit (bwt_pos, seqID, meta) tuples.
			sa_lookup_prepass_emit_kernel <<< batch_size, SAL_SORT_BLOCKDIMX, 0, stream >>> (
					proc->d_opt, proc->d_seq, proc->d_seq_offset,
					proc->d_aux, proc->d_read_base_offsets,
					proc->d_sal_keys, proc->d_sal_vals,
					(sal_meta_t*)proc->d_sal_meta_buf);
			CUDA_CHECK_KERNEL_RUN();

			// Step 5: sort (d_sal_keys, d_sal_vals) by bwt_pos.
			//   d_sal_vals_sorted (d_sal_perm) maps sorted position → seqID.
			{
				void  *d_tmp   = proc->d_gss_cub_temp;
				size_t tmp_bytes = proc->d_gss_cub_temp_bytes;
				CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
						d_tmp, tmp_bytes,
						proc->d_sal_keys, proc->d_sal_keys_sorted,
						proc->d_sal_vals,  proc->d_sal_perm,
						batch_total_seeds, 0, sizeof(uint64_t) * 8, stream));
			}
			CUDA_CHECK_KERNEL_RUN();

			// Step 6: scatter meta via perm so d_sal_meta_sorted[i] = meta[perm[i]].
			{
				sal_meta_t *d_meta_orig   = (sal_meta_t*)proc->d_sal_meta_buf;
				sal_meta_t *d_meta_sorted = d_meta_orig + proc->gss_max_seeds;
				int grid = (batch_total_seeds + SAL_SORT_BLOCKDIMX - 1) / SAL_SORT_BLOCKDIMX;
				// Inline meta scatter: d_meta_sorted[i] = d_meta_orig[perm[i]]
				// (dedicated kernel declared below as sal_meta_scatter_kernel).
				sal_meta_scatter_kernel <<< grid, SAL_SORT_BLOCKDIMX, 0, stream >>> (
						d_meta_orig, d_meta_sorted, proc->d_sal_perm, batch_total_seeds);
				CUDA_CHECK_KERNEL_RUN();
			}

			// Step 7: sorted SA lookup.
			{
				int grid = (batch_total_seeds + SAL_SORT_BLOCKDIMX - 1) / SAL_SORT_BLOCKDIMX;
				sa_lookup_gss_kernel <<< grid, SAL_SORT_BLOCKDIMX, 0, stream >>> (
						proc->d_fmi, proc->d_sal_keys_sorted, proc->d_sal_rbeg,
						batch_total_seeds);
				CUDA_CHECK_KERNEL_RUN();
			}

			// Step 8: allocate d_seq_seeds[seqID].a from pool and reset actual counters.
			// Use a dedicated setup kernel that mirrors the alloc done in sa_lookup_kernel.
			{
				int grid = (batch_size + SAL_SORT_BLOCKDIMX - 1) / SAL_SORT_BLOCKDIMX;
				sal_alloc_seeds_kernel <<< grid, SAL_SORT_BLOCKDIMX, 0, stream >>> (
						proc->d_per_read_total_seeds, proc->d_seq_seeds,
						proc->d_seq_actual_count, proc->d_buffer_pools, batch_size);
				CUDA_CHECK_KERNEL_RUN();
			}

			// Step 9: scatter results back to d_seq_seeds.
			{
				sal_meta_t *d_meta_sorted = (sal_meta_t*)proc->d_sal_meta_buf + proc->gss_max_seeds;
				int grid = (batch_total_seeds + SAL_SORT_BLOCKDIMX - 1) / SAL_SORT_BLOCKDIMX;
				sa_lookup_scatter_kernel <<< grid, SAL_SORT_BLOCKDIMX, 0, stream >>> (
						proc->d_fmi, proc->d_bns,
						proc->d_sal_rbeg, proc->d_sal_perm,
						d_meta_sorted, batch_total_seeds,
						proc->d_read_base_offsets, batch_size,
						proc->d_seq_seeds, proc->d_seq_actual_count);
				CUDA_CHECK_KERNEL_RUN();
			}

			// Step 10: write final n counts from actual counters.
			{
				int grid = (batch_size + SAL_SORT_BLOCKDIMX - 1) / SAL_SORT_BLOCKDIMX;
				sal_finalize_counts_kernel <<< grid, SAL_SORT_BLOCKDIMX, 0, stream >>> (
						proc->d_seq_actual_count, proc->d_seq_seeds, batch_size);
				CUDA_CHECK_KERNEL_RUN();
			}
		} else {
			// === Original per-read path (E. coli / small batches / fallback) ===
			sa_lookup_kernel <<< batch_size, SAL_SORT_BLOCKDIMX, 0, stream >>> (
					proc->d_opt, proc->d_fmi, proc->d_bns,
					proc->d_seq, proc->d_seq_offset, proc->d_aux,
					proc->d_seq_seeds, proc->d_buffer_pools);
			CUDA_CHECK_KERNEL_RUN();
		}
		} else {
			// maybe_use_gss=false: read length < SAL_GSS_MIN_READ_LEN.
			// Skip prepass entirely; go straight to original per-read path.
			sa_lookup_kernel <<< batch_size, SAL_SORT_BLOCKDIMX, 0, stream >>> (
					proc->d_opt, proc->d_fmi, proc->d_bns,
					proc->d_seq, proc->d_seq_offset, proc->d_aux,
					proc->d_seq_seeds, proc->d_buffer_pools);
			CUDA_CHECK_KERNEL_RUN();
		}
	}
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][C_SAL] += step_lap;
	if(PRINT(_SEED) || PRINT(_ALL_CHAINING)){
		for(int readID = 0; readID < batch_size; readID++)
		{
			printSeed<<<1, WARPSIZE, 0, stream>>>(proc->d_seq_seeds, readID, _SEED);
			CUDA_CHECK_KERNEL_RUN();
		}
	}
	CUDA_TIMER_START(step_lap);
	// sort translated seeds in rbeg order
	sort_seeds_low 
		<<< batch_size, SORTSEEDSLOW_BLOCKDIMX, 0, stream >>> (
				proc->d_seq_seeds,
				proc->d_buffer_pools);
	CUDA_CHECK_KERNEL_RUN();
	sort_seeds_high 
		<<< batch_size, SORTSEEDSHIGH_BLOCKDIMX, 0, stream >>> (
				proc->d_seq_seeds,
				proc->d_buffer_pools);
	CUDA_CHECK_KERNEL_RUN();
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][C_SORT_SEEDS] += step_lap;
	if(PRINT(_STSEED) || PRINT(_ALL_CHAINING)){
		for(int readID = 0; readID < batch_size; readID++)
		{
			printSeed<<<1, WARPSIZE, 0, stream>>>(proc->d_seq_seeds, readID, _STSEED);
			CUDA_CHECK_KERNEL_RUN();
		}
	}
	CUDA_TIMER_START(step_lap);
	chain_seeds_parallel
		<<< batch_size, CHAIN_PARALLEL_BLOCKDIMX, 0, stream >>>(
				batch_size,
				proc->d_opt, proc->d_bns,
				proc->d_seq, proc->d_seq_offset, proc->d_seq_seeds,
				proc->d_chains,
				proc->d_buffer_pools);
	CUDA_CHECK_KERNEL_RUN();
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][C_CHAIN] += step_lap;
	if(PRINT(_CHAIN) || PRINT(_ALL_CHAINING)){
		for(int readID = 0; readID < batch_size; readID++){
			printChain<<<1, WARPSIZE, 0, stream>>>(proc->d_chains, readID, _CHAIN);
			CUDA_CHECK_KERNEL_RUN();
		}
	}
	// sort before filtering
	CUDA_TIMER_START(step_lap);
	sort_chains_by_weight 
		<<< batch_size, SORTCHAIN_BLOCKDIMX, 
		MAX_N_CHAIN*2*sizeof(uint16_t)+sizeof(mem_chain_t**), stream>>>
			(proc->d_chains, proc->d_buffer_pools);
	CUDA_CHECK_KERNEL_RUN();
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][C_SORT_CHAINS] += step_lap;
	if(PRINT(_STCHAIN) || PRINT(_ALL_CHAINING)){
		for(int readID = 0; readID < batch_size; readID++)
		{
			printChain<<<1, WARPSIZE, 0, stream>>>(proc->d_chains, readID, _STCHAIN);
			CUDA_CHECK_KERNEL_RUN();
		}
	}
	// filter chains
	CUDA_TIMER_START(step_lap);
	filter_chains <<< batch_size, CHAIN_FLT_BLOCKSIZE, MAX_N_CHAIN*(3*sizeof(uint16_t)+sizeof(uint8_t))+8*sizeof(int), stream >>> (
			proc->d_opt, 
			proc->d_chains, 	// input and output
			proc->d_buffer_pools);
	CUDA_CHECK_KERNEL_RUN();
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][C_FILTER] += step_lap;
	if(PRINT(_FTCHAIN) || PRINT(_ALL_CHAINING)){
		for(int readID = 0; readID < batch_size; readID++)
		{
			printChain<<<1, WARPSIZE, 0, stream>>>(proc->d_chains, readID, _FTCHAIN);
			CUDA_CHECK_KERNEL_RUN();
		}
	}
	// Filter low-quality seeds within chains (matches bwa-mem2's mem_flt_chained_seeds).
	// Block-per-read: FCS_BLOCKDIM threads score seeds in parallel.
	CUDA_TIMER_START(step_lap);
	filter_chained_seeds <<< batch_size, FCS_BLOCKDIM, 0, stream >>> (
			proc->d_opt, proc->d_bns, proc->d_pac,
			proc->d_seq, proc->d_seq_offset,
			proc->d_chains, batch_size,
			proc->d_buffer_pools);
	CUDA_CHECK_KERNEL_RUN();
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][E_FLT_SEEDS] += step_lap;

	// (3/3) Extending
	// Extending -> Local extending (1/2)
	CUDA_TIMER_START(step_lap);
	sw_seed_prep <<< ceil((float)batch_size / WARPSIZE) == 0 ? 1 : ceil((float)batch_size / WARPSIZE), WARPSIZE, 0, stream >>>(
			proc->d_chains, proc->d_regs,
		       	proc->d_seed_records, proc->d_Nseeds, batch_size,
			proc->d_buffer_pools
			);
	CUDA_CHECK_KERNEL_RUN();
	// d_Nseeds is host-mapped: synchronize the stream then read directly from host pointer.
	CUDA_CHECK(cudaStreamSynchronize(stream));
	num_seeds_to_extend = *proc->h_Nseeds;

	if(num_seeds_to_extend==0){
		// n_processed managed by dispatcher (pipeline.cpp) — do not increment here.
		return 0;
	}
	// Fused pair-gen + SW extension (replaces separate extend_pair_generate + local_extend).
	// Eliminates 80 B/seed global round-trip through d_seed_records.
	// d_seed_records allocation is kept for kernel test harness compatibility.
	if(PRINT(_DETAIL)){
		std::cerr << "GPU " << gpuid << "  " << "# local extending pairs: " << num_seeds_to_extend << std::endl;
	}
	CUDA_TIMER_START(step_lap);  // charge fused kernel to E_PAIRGEN+E_EXTEND combined (logged to E_EXTEND below)
	extend_and_sw <<< num_seeds_to_extend, WARPSIZE, 0, stream >>> (
			proc->d_opt, proc->d_bns, proc->d_pac, proc->d_seq, proc->d_seq_offset,
			proc->d_chains, proc->d_regs, proc->d_seed_records, proc->d_Nseeds,
			batch_size, proc->d_buffer_pools
			);
	CUDA_CHECK_KERNEL_RUN();
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][E_PAIRGEN] += 0;   // fused; no separate pair-gen time
	tprof[gpuid][E_EXTEND]  += step_lap;
	if(PRINT(_REGION) || PRINT(_ALL_EXTENDING)){
		for(int readID = 0; readID < batch_size; readID++)
		{
			printReg<<<1, WARPSIZE, 0, stream>>>(proc->d_regs, readID, _REGION);
			CUDA_CHECK_KERNEL_RUN();
		}
	}
	// filter contained regions: purge seeds contained in higher-scoring regions
	// (port of bwa-mem2's post-extension seed containment check)
	// Block-cooperative (one block per read, SORT_REGIONS_BLOCK threads,
	// parallel CUB sort by score DESC + parallel OR-reduce for containment).
	CUDA_TIMER_START(step_lap);
	filter_contained_regions <<< batch_size, SORT_REGIONS_BLOCK, FCR_SM_BYTES, stream >>> (
			proc->d_opt, proc->d_chains,
			proc->d_regs, proc->d_seed_records, proc->d_Nseeds,
			proc->d_seq_offset, batch_size);
	CUDA_CHECK_KERNEL_RUN();
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][E_FCR] += step_lap;

	// Stage-level arena reset: seeding/chaining pools (0-31) are no longer needed.
	// d_seq_seeds[i].a, B-tree nodes, chain seed arrays, sort scratch are all dead after
	// local_extend + filter_contained_regions.  Extension/traceback data lives in pools 32-63.
	capturePoolPeak(proc->d_buffer_pools, 0, 32, stream);  // G3_POOL_PROFILE: capture before reset
	CUDAResetBufferPoolRange(proc->d_buffer_pools, 0, 32, stream);

	// patch regions: merge/dedup colinear regions (port of bwa-mem2 mem_sort_dedup_patch)
	// Block-cooperative (one block per read, SORT_REGIONS_BLOCK threads, parallel
	// CUB sorts + parallel dedup with serial-fallback for merge cases).
	CUDA_TIMER_START(step_lap);
	patch_regions <<< batch_size, SORT_REGIONS_BLOCK, PATCH_REGIONS_SM_BYTES, stream >>> (
			proc->d_opt, proc->d_bns, proc->d_pac,
			proc->d_seq, proc->d_seq_offset,
			proc->d_regs, proc->d_buffer_pools,
			batch_size
			);
	CUDA_CHECK_KERNEL_RUN();
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][E_PATCH] += step_lap;

	// Fused filter_regions + apply_score_filter.
	// Single pass: (1) set is_alt flags (all 256 threads), (2) compact score<T
	// regions (thread 0).  Runs BEFORE sort_regions so is_alt is ready for the
	// two-pass sort.  Replaces filter_regions (320t) + apply_score_filter (32t,
	// thread-0-only); originals kept in region.cu for the kernel_test harness.
	CUDA_TIMER_START(step_lap);
	filter_and_score_threshold <<< batch_size, 256, 0, stream >>>(
			proc->d_opt, proc->d_bns, proc->d_regs, proc->n_processed);
	CUDA_CHECK_KERNEL_RUN();
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][E_FILTER_MARK] += step_lap;
	if(PRINT(_FTREGION) || PRINT(_ALL_EXTENDING)){
		for(int readID = 0; readID < batch_size; readID++)
		{
			printReg<<<1, WARPSIZE, 0, stream>>>(proc->d_regs, readID, _FTREGION);
			CUDA_CHECK_KERNEL_RUN();
		}
	}
	// sort regions before traceback (one block per read, 256 threads, parallel sort+marking)
	CUDA_TIMER_START(step_lap);
	sort_regions <<< batch_size, SORT_REGIONS_BLOCK, SORT_REGIONS_SM_BYTES, stream >>>(
			proc->d_opt, proc->d_regs, batch_size, proc->n_processed, proc->d_buffer_pools);
	CUDA_CHECK_KERNEL_RUN();
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][E_SORT_ALNS] += step_lap;
	if(PRINT(_STREGION) || PRINT(_ALL_EXTENDING)){
		for(int readID = 0; readID < batch_size; readID++)
		{
			printReg<<<1, WARPSIZE, 0, stream>>>(proc->d_regs, readID, _STREGION);
			CUDA_CHECK_KERNEL_RUN();
		}
	}
	// compute offsets to the result alignments array.
	CUDA_TIMER_START(step_lap);
	compute_offsets_on_host(proc, batch_size, stream);
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][E_OFFSETS] += step_lap;

	// Extending -> Traceback (2/2)
	CUDA_TIMER_START(step_lap);
	// d_Nalns is host-mapped: reset directly through the host pointer (no GPU round-trip)
	*proc->h_Nalns = 0;
	// preproc
	finalize_prep1 <<< ceil((float)batch_size / WARPSIZE) == 0 ? 1 : ceil((float)batch_size / WARPSIZE), WARPSIZE, 0, stream >>>(
			batch_size,
			proc->d_regs, proc->d_alns, proc->d_seed_records, proc->d_Nalns,
			proc->d_buffer_pools);
	CUDA_CHECK_KERNEL_RUN();

	// d_Nalns is host-mapped: synchronize the stream then read directly from host pointer.
	CUDA_CHECK(cudaStreamSynchronize(stream));
	batch_num_alns = *proc->h_Nalns;
	if(batch_num_alns > MAX_ALN_CNT){
		std::cerr << "[ERROR] batch_num_alns=" << batch_num_alns
		          << " exceeds MAX_ALN_CNT=" << MAX_ALN_CNT
		          << " — increase MAX_ALN_CNT in macro.h\n";
		exit(EXIT_FAILURE);
	}
	if(batch_num_alns==0){
		// n_processed managed by dispatcher (pipeline.cpp) — do not increment here.
		return 0;
	}
	// preproc
	finalize_prep2 <<< ceil((float)batch_num_alns / WARPSIZE) == 0 ? 1 : ceil((float)batch_num_alns / WARPSIZE), WARPSIZE, 0, stream >>>(
			proc->d_opt, proc->d_seq, proc->d_seq_offset,
			proc->d_pac, proc->d_bns,
			proc->d_regs, proc->d_alns, proc->d_seed_records, batch_num_alns,
			proc->d_sortkeys_in,	// sortkeys_in = bandwidth * rlen
			proc->d_seqIDs_in,
			proc->d_buffer_pools);
	CUDA_CHECK_KERNEL_RUN();
	// preproc: sort (use preallocated CUB temp storage — avoids per-batch cudaMalloc/cudaFree)
	d_temp_storage    = proc->d_cub_temp;
	temp_storage_size = proc->d_cub_temp_bytes;
	CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_size, proc->d_sortkeys_in, proc->d_sortkeys_out, proc->d_seqIDs_in, proc->d_seqIDs_out, batch_num_alns, 0, 8*sizeof(int), stream));
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][E_T_PAIRGEN] += step_lap;
	if(PRINT(_TBPAIR) || PRINT(_ALL_EXTENDING)){
		printPair<<<1, WARPSIZE, 0, stream>>>(proc->d_seed_records, batch_num_alns, _TBPAIR);
		CUDA_CHECK_KERNEL_RUN();
	}

	if(PRINT(_DETAIL)){
		std::cerr << "GPU " << gpuid << "  " << "# traceback pairs: " << batch_num_alns << std::endl;
	}
	// traceback + finalize: fused warp-cooperative kernel (one block per alignment)
	CUDA_TIMER_START(step_lap);
	traceback_and_finalize<<< batch_num_alns == 0 ? 1 : batch_num_alns, WARPSIZE, 0, stream >>>(
			proc->d_opt, proc->d_bns, proc->d_seq, proc->d_seq_offset,
			proc->d_regs, proc->d_alns,
			proc->d_seed_records, batch_num_alns, proc->d_seqIDs_out,
			proc->d_buffer_pools,
			proc->d_offsets,
			proc->d_rids,
			proc->d_positions,
			proc->d_ncigars,
			proc->d_cigars,
			proc->d_flags,
			proc->d_mapqs
			);
	CUDA_CHECK_KERNEL_RUN();
	if(PRINT(_RESULT) || PRINT(_ALL_EXTENDING)){
		for(int readID = 0; readID < batch_size; readID++)
		{
			printAln<<<1, WARPSIZE, 0, stream>>>(proc->d_bns, proc->d_alns, readID, _RESULT);
			CUDA_CHECK_KERNEL_RUN();
		}
	}
	capturePoolPeak(proc->d_buffer_pools, 32, 64, stream);  // G3_POOL_PROFILE: ext/traceback peak
	CUDA_TIMER_END(step_lap);
	tprof[gpuid][E_TRACEBACK] += step_lap;
	// Explicit stream sync when G3_KERNEL_TIMING is off (CUDA_TIMER_END is then a no-op).
	// Ensures d_positions/d_ncigars/etc. are host-visible before memcpy_output reads them.
	if (!g_kernel_timing)
		CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_TIMER_DESTROY;
	return 0;
}
