/*
 * chain.cu -- Seed sorting, chaining, chain sorting, and chain filtering kernels.
 */

/*
 * Portions of this file are adapted from minhhpham/bwa
 * (https://github.com/minhhpham/bwa), Copyright (c) minhhpham,
 * GPL-3.0 licensed — a GPU port of BWA-MEM (Copyright (c) Dana-Farber
 * Cancer Institute, Broad Institute, Genome Research Ltd.). This file is
 * part of G3SA and is licensed under GPL-3.0 (see LICENSE).
 */

#include "gpu_types.h"
#include "gmem_alloc.cuh"
#include "bntseq.h"
#include "ksw.cuh"
#include <string.h>
#include "cuda_wrapper.h"
#include "macro.h"
#include "seed.cuh"

#include <cub/cub.cuh>

#include "chain.cuh"


/********************
 * Filtering chains *
 ********************/

__device__ int mem_chain_weight(const mem_chain_t *c)
{
    int64_t end;
    int j, w = 0, tmp;
    for (j = 0, end = 0; j < c->n; ++j) {
        const mem_seed_t *s = &c->seeds[j];
        if (s->qbeg >= end) w += s->len;
        else if (s->qbeg + s->len > end) w += s->qbeg + s->len - end;
        end = end > s->qbeg + s->len? end : s->qbeg + s->len;
    }
    tmp = w; w = 0;
    for (j = 0, end = 0; j < c->n; ++j) {
        const mem_seed_t *s = &c->seeds[j];
        if (s->rbeg >= end) w += s->len;
        else if (s->rbeg + s->len > end) w += s->rbeg + s->len - end;
        end = end > s->rbeg + s->len? end : s->rbeg + s->len;
    }
    w = w < tmp? w : tmp;
    return w < 1<<30? w : (1<<30)-1;
}

/*********************************
 * Test if a seed is good enough *
 *********************************/

__device__ int mem_seed_sw(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, int l_query, const uint8_t *query, const mem_seed_t *s, void* d_buffer_ptr)
{
    int qb, qe, rid;
    int64_t rb, re, mid, l_pac = bns->l_pac;
    uint8_t *rseq = 0;
    kswr_t x;

    if (s->len >= MEM_SHORT_LEN) return -1; // the seed is longer than the max-extend; no need to do SW
    qb = s->qbeg, qe = s->qbeg + s->len;
    rb = s->rbeg, re = s->rbeg + s->len;
    mid = (rb + re) >> 1;
    qb -= MEM_SHORT_EXT; qb = qb > 0? qb : 0;
    qe += MEM_SHORT_EXT; qe = qe < l_query? qe : l_query;
    rb -= MEM_SHORT_EXT; rb = rb > 0? rb : 0;
    re += MEM_SHORT_EXT; re = re < l_pac<<1? re : l_pac<<1;
    if (rb < l_pac && l_pac < re) {
        if (mid < l_pac) re = l_pac;
        else rb = l_pac;
    }
    if (qe - qb >= MEM_SHORT_LEN || re - rb >= MEM_SHORT_LEN) return -1; // the seed seems good enough; no need to do SW

    /* Make a private copy of the query window before calling ksw_align2.
     * ksw_align2 (with KSW_XSTART) temporarily reverses the query sequence
     * in-place to find the alignment start position.  When filter_chained_seeds
     * is parallelised (multiple threads scoring seeds for the same read
     * simultaneously), threads sharing overlapping query windows would race
     * on the global d_seq array.  A private copy is always correct and has
     * negligible cost (< 200 bytes for MEM_SHORT_LEN-bounded seeds). */
    int qlen_sw = qe - qb;
    uint8_t *query_copy = (uint8_t*)CUDAKernelMalloc(d_buffer_ptr, qlen_sw, 1);
    for (int i = 0; i < qlen_sw; i++) query_copy[i] = query[qb + i];

    rseq = bns_fetch_seq_gpu(bns, pac, &rb, mid, &re, &rid, d_buffer_ptr);
    x = ksw_align2(qlen_sw, query_copy, re - rb, rseq, 5, opt->mat, opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, KSW_XSTART, d_buffer_ptr);
    return x.score;
}


/* for each read, sort seeds by len ASC, then by qbeg ASC
   use cub::blockRadixSort
 */
// process reads who have less seeds
__global__ void sort_seeds_low(
        mem_seed_v *d_seq_seeds,
        void *d_buffer_pools
        )
{
    // seqID = blockIdx.x
    int n_seeds = d_seq_seeds[blockIdx.x].n;
    if (n_seeds==0) return;
    if (n_seeds>SORTSEEDSLOW_MAX_NSEEDS) return;

    mem_seed_t *seed_arrA = d_seq_seeds[blockIdx.x].a;

    // Specialize BlockRadixSort
    typedef cub::BlockRadixSort<int64_t, SORTSEEDSLOW_BLOCKDIMX, SORTSEEDSLOW_NKEYS_THREAD, int> BlockRadixSort;
    // Allocate shared mem
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // Block sort variables
    int64_t thread_keys[SORTSEEDSLOW_NKEYS_THREAD];
    int thread_values[SORTSEEDSLOW_NKEYS_THREAD];
    int old_pos, new_pos;

    __shared__ mem_seed_t* s_seed_arrB; // new bucket
    if (threadIdx.x==0){
        void *d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, blockIdx.x%32);
        s_seed_arrB = (mem_seed_t*)CUDAKernelMalloc(d_buffer_ptr, n_seeds*sizeof(mem_seed_t), 8);
    }
    __syncthreads(); __syncwarp();

    // (1/2) sort by (effectively qe) (len)
    for(int i=0; i<SORTSEEDSLOW_NKEYS_THREAD; i++){
        old_pos = threadIdx.x*SORTSEEDSLOW_NKEYS_THREAD+i;
        if(old_pos < n_seeds){
            thread_keys[i] = seed_arrA[old_pos].len;
            thread_values[i] = old_pos;
        } else{ // pad with INT64_MAX
            thread_keys[i] = INT64_MAX;
            thread_values[i] = -1;
        }
    }
    BlockRadixSort(temp_storage).Sort(thread_keys, thread_values); // it is stable

    for(int i=0; i<SORTSEEDSLOW_NKEYS_THREAD; i++){ //reorder
        new_pos = threadIdx.x * SORTSEEDSLOW_NKEYS_THREAD + i;
        if(new_pos >= n_seeds) break;
        if(thread_values[i]==-1){
            printf("Error: sorting result incorrect. SeqID=%d\n", blockIdx.x);
            __trap();
        }
        s_seed_arrB[new_pos] = seed_arrA[thread_values[i]];
    }
    __syncthreads(); __syncwarp();
    __syncwarp();

    // (2/2) sort by qb
    for(int i=0; i<SORTSEEDSLOW_NKEYS_THREAD; i++){
        old_pos = threadIdx.x*SORTSEEDSLOW_NKEYS_THREAD+i;
        if(old_pos < n_seeds){
            thread_keys[i] = s_seed_arrB[old_pos].qbeg;
            thread_values[i] = old_pos;
        } else{ // pad with INT64_MAX
            thread_keys[i] = INT64_MAX;
            thread_values[i] = -1;
        }
    }
    BlockRadixSort(temp_storage).Sort(thread_keys, thread_values); // it is stable

    for(int i=0; i<SORTSEEDSLOW_NKEYS_THREAD; i++){ //reorder
        new_pos = threadIdx.x * SORTSEEDSLOW_NKEYS_THREAD + i;
        if(new_pos >= n_seeds) break;
        if(thread_values[i]==-1){
            printf("Error: sorting result incorrect. SeqID=%d\n", blockIdx.x);
            __trap();
        }
        seed_arrA[new_pos] = s_seed_arrB[thread_values[i]];
    }
}


// process reads who have more seeds
__global__ void sort_seeds_high(
        mem_seed_v *d_seq_seeds,
        void *d_buffer_pools
        )
{
    // seqID = blockIdx.x
    int n_seeds = d_seq_seeds[blockIdx.x].n;
    if (n_seeds<=SORTSEEDSLOW_MAX_NSEEDS) return;

    mem_seed_t *seed_arrA = d_seq_seeds[blockIdx.x].a;

    // Specialize BlockRadixSort
    typedef cub::BlockRadixSort<int64_t, SORTSEEDSHIGH_BLOCKDIMX, SORTSEEDSHIGH_NKEYS_THREAD, int> BlockRadixSort;
    // Allocate shared mem
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // Block sort variables
    int64_t thread_keys[SORTSEEDSHIGH_NKEYS_THREAD];
    int thread_values[SORTSEEDSHIGH_NKEYS_THREAD];
    int old_pos, new_pos;

    __shared__ mem_seed_t* s_seed_arrB; // new bucket
    if (threadIdx.x==0){
        void *d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, blockIdx.x%32);
        s_seed_arrB = (mem_seed_t*)CUDAKernelMalloc(d_buffer_ptr, n_seeds*sizeof(mem_seed_t), 8);
    }
    __syncthreads(); __syncwarp();
    __syncwarp();

    // (1/2) sort by qe
    for(int i=0; i<SORTSEEDSHIGH_NKEYS_THREAD; i++){
        old_pos = threadIdx.x*SORTSEEDSHIGH_NKEYS_THREAD+i;
        if(old_pos < n_seeds){
            thread_keys[i] = seed_arrA[old_pos].qbeg + seed_arrA[old_pos].len;
            thread_values[i] = old_pos;
        } else{ // pad with INT64_MAX
            thread_keys[i] = INT64_MAX;
            thread_values[i] = -1;
        }
    }
    BlockRadixSort(temp_storage).Sort(thread_keys, thread_values); // it is stable

    for(int i=0; i<SORTSEEDSHIGH_NKEYS_THREAD; i++){ //reorder
        new_pos = threadIdx.x * SORTSEEDSHIGH_NKEYS_THREAD + i;
        if(new_pos >= n_seeds) break;
        if(thread_values[i]==-1){
            printf("Error: sorting result incorrect. SeqID=%d\n", blockIdx.x);
            __trap();
        }
        s_seed_arrB[new_pos] = seed_arrA[thread_values[i]];
    }
    __syncthreads(); __syncwarp();
    __syncwarp();

    // (2/2) sort by qb
    for(int i=0; i<SORTSEEDSHIGH_NKEYS_THREAD; i++){
        old_pos = threadIdx.x*SORTSEEDSHIGH_NKEYS_THREAD+i;
        if(old_pos < n_seeds){
            thread_keys[i] = s_seed_arrB[old_pos].qbeg;
            thread_values[i] = old_pos;
        } else{ // pad with INT64_MAX
            thread_keys[i] = INT64_MAX;
            thread_values[i] = -1;
        }
    }
    BlockRadixSort(temp_storage).Sort(thread_keys, thread_values); // it is stable

    for(int i=0; i<SORTSEEDSHIGH_NKEYS_THREAD; i++){ //reorder
        new_pos = threadIdx.x * SORTSEEDSHIGH_NKEYS_THREAD + i;
        if(new_pos >= n_seeds) break;
        if(thread_values[i]==-1){
            printf("Error: sorting result incorrect. SeqID=%d\n", blockIdx.x);
            __trap();
        }
        seed_arrA[new_pos] = s_seed_arrB[thread_values[i]];
    }
}


/*
 * chain_seeds_parallel — Block-per-read parallel chaining (replaces legacy B-tree path).
 *
 * Seeds are pre-sorted by (qbeg ASC, len ASC) by sort_seeds_low/high.
 * Each CUDA thread handles one or more seeds (stride loop over j).
 * For each seed j, the thread scans backward from j-1 to 0 and finds the
 * seed i with maximum rbeg ≤ j.rbeg that passes all chaining conditions.
 * atomicMin on S_succ[i] resolves conflicts when multiple j's claim the same i.
 * Thread 0 builds the final chain structs from the predecessor/successor arrays.
 *
 * Approximates bwa-mem2's sequential B-tree chaining:
 * - Correct for single-seed chains (exact match).
 * - For multi-seed chains: uses each seed's rbeg rather than chain.pos (first seed's
 *   rbeg), which is the comparison point used inside test_and_merge anyway.
 * - Conflict resolution ensures at most one successor per seed, matching the
 *   property that each B-tree chain absorbs at most one new seed per step.
 */
__global__ void chain_seeds_parallel(
        int batch_size,
        const mem_opt_t *d_opt,
        const bntseq_t *d_bns,
        const uint8_t *d_seq,
        int *d_seq_offset,
        mem_seed_v *d_seq_seeds,
        mem_chain_v *d_chains,
        void *d_buffer_pools
        )
{
    int seqID = blockIdx.x;
    if (seqID >= batch_size) return;

    int n_seeds = d_seq_seeds[seqID].n;
    if (n_seeds == 0) {
        if (threadIdx.x == 0) d_chains[seqID].n = 0;
        return;
    }
    if (n_seeds > CHAIN_PARALLEL_MAX_NSEEDS)
        n_seeds = CHAIN_PARALLEL_MAX_NSEEDS;

    mem_seed_t *seed_a = d_seq_seeds[seqID].a;
    const int max_chain_gap = d_opt->max_chain_gap;
    const int bandwidth_gap = d_opt->w;
    const int64_t l_pac = d_bns->l_pac;
    const int l_seq = d_seq_offset[seqID + 1] - d_seq_offset[seqID];

    /* Shared arrays indexed [0, n_seeds).
     * S_preceding[j]: direct predecessor of j (= j if chain head, -1 if invalid).
     * S_succ[j]:      first successor of j (INT_MAX if none). */
    __shared__ int16_t S_preceding[CHAIN_PARALLEL_MAX_NSEEDS];
    __shared__ int     S_succ[CHAIN_PARALLEL_MAX_NSEEDS];

    for (int i = threadIdx.x; i < n_seeds; i += blockDim.x) {
        S_preceding[i] = (int16_t)i;   // default: chain head
        S_succ[i]      = INT_MAX;       // no successor yet
    }
    __syncthreads();

    /* Phase 1: parallel predecessor search.
     * Each thread independently searches for the best predecessor for its seeds. */
    for (int j = threadIdx.x + 1; j < n_seeds; j += blockDim.x) {
        if (seed_a[j].rid < 0) {
            S_preceding[j] = -1;
            continue;
        }

        const int64_t rbeg_j = seed_a[j].rbeg;
        const int     qbeg_j = seed_a[j].qbeg;
        const int     rid_j  = seed_a[j].rid;
        int64_t best_rbeg = -1;
        int     best_i    = j;   // j → self means chain head

        for (int i = j - 1; i >= 0; i--) {
            /* Early termination: seeds are qbeg-sorted; once the query gap
             * exceeds max_chain_gap + l_seq (an upper bound on seed length),
             * no earlier seed can satisfy x - len_i < max_chain_gap. */
            int x = qbeg_j - seed_a[i].qbeg;
            if (x > max_chain_gap + l_seq) break;

            if (seed_a[i].rid < 0) continue;

            const int64_t rbeg_i = seed_a[i].rbeg;

            /* Predecessor must precede j in reference space. */
            if (rbeg_i > rbeg_j) continue;

            /* Reference gap check: y - len_i < max_chain_gap. */
            int64_t y = rbeg_j - rbeg_i;
            if (y - seed_a[i].len >= max_chain_gap) continue;

            /* Query gap check: x - len_i < max_chain_gap. */
            if (x - seed_a[i].len >= max_chain_gap) continue;

            /* Bandwidth check: |x - y| ≤ w. */
            int64_t diff = x - y;
            if (diff > bandwidth_gap || -diff > bandwidth_gap) continue;

            /* Strand check: don't chain across the l_pac midpoint. */
            if ((rbeg_i < l_pac) != (rbeg_j < l_pac)) continue;

            /* Chromosome check. */
            if (seed_a[i].rid != rid_j) continue;

            /* Valid predecessor — take it if nearest (maximum rbeg ≤ rbeg_j). */
            if (rbeg_i > best_rbeg) {
                best_rbeg = rbeg_i;
                best_i    = i;
            }
            /* Keep scanning: a later seed (smaller i) might have a larger rbeg
             * that is still ≤ rbeg_j and thus a better (nearer) predecessor. */
        }

        S_preceding[j] = (int16_t)best_i;
        if (best_i != j)
            atomicMin(&S_succ[best_i], j);
    }
    __syncthreads();

    /* Phase 2: conflict resolution.
     * If two threads both chose seed i as best predecessor, atomicMin gives
     * S_succ[i] = min(j1, j2).  The loser (larger index) becomes a chain head. */
    for (int j = threadIdx.x + 1; j < n_seeds; j += blockDim.x) {
        int prec = (int)S_preceding[j];
        if (prec < 0 || prec == j) continue;
        if (S_succ[prec] != j)
            S_preceding[j] = (int16_t)j;   // lost conflict → new chain head
    }
    __syncthreads();

    /* Phase 3: build chain structs — block-cooperative.
     *
     * Phase 3a: all 256 threads stride over seeds to identify chain heads.
     *   CUB BlockScan::ExclusiveSum gives each head its index in chain_a[].
     *   Thread 0 reads the total chain count and allocates chain_a[].
     *
     * Phase 3b: each thread owns chains whose head seed index it covered in
     *   the stride loop.  Each thread walks its chains' successor lists
     *   independently (count → allocate seed array → fill), using its own
     *   pool shard to avoid allocator contention.
     *
     * No early return before the CUB scan — all threads must participate.
     */

    /* Shared state shared across sub-phases. */
    typedef cub::BlockScan<int, CHAIN_PARALLEL_BLOCKDIMX> BlockScan3;
    __shared__ typename BlockScan3::TempStorage scan3_storage;
    __shared__ mem_chain_t *s_chain_a;
    __shared__ int          s_n_chains;

    /* ---- Phase 3a: parallel chain-head count via BlockScan ---- */

    /* Each thread accumulates is_head for its strided seeds. */
    int my_head_count = 0;
    for (int j = threadIdx.x; j < n_seeds; j += blockDim.x)
        if ((int)S_preceding[j] == j && seed_a[j].rid >= 0) my_head_count++;

    /* Exclusive prefix sum: thread t's scan_offset = number of heads owned
     * by threads 0..t-1 across their full strides. */
    int scan_offset;
    int block_total;
    BlockScan3(scan3_storage).ExclusiveSum(my_head_count, scan_offset, block_total);
    /* block_total = total chain count across all threads. */

    if (threadIdx.x == 0) {
        s_n_chains = block_total;
        if (block_total > 0) {
            void *d_buf0 = CUDAKernelSelectPool(d_buffer_pools, blockIdx.x % 32);
            s_chain_a = (mem_chain_t*)CUDAKernelMalloc(
                    d_buf0, block_total * sizeof(mem_chain_t), 8);
        } else {
            s_chain_a = nullptr;
        }
    }
    __syncthreads();

    if (s_n_chains == 0) {
        if (threadIdx.x == 0) d_chains[seqID].n = 0;
        return;
    }

    mem_chain_t *chain_a = s_chain_a;

    /* ---- Phase 3b: parallel chain-struct fill ---- */

    /* Each thread uses its own pool shard to avoid allocator contention. */
    void *d_buffer_ptr = CUDAKernelSelectPool(
            d_buffer_pools, (blockIdx.x * CHAIN_PARALLEL_BLOCKDIMX + threadIdx.x) % 32);

    /* Stride over seeds; for each chain head this thread owns, walk the
     * successor chain and fill the chain struct + seed array.
     * scan_offset is this thread's starting chain index in chain_a[]. */
    int local_chain_idx = 0;
    for (int head = threadIdx.x; head < n_seeds; head += blockDim.x) {
        if ((int)S_preceding[head] != head || seed_a[head].rid < 0) continue;

        int chain_a_idx = scan_offset + local_chain_idx;
        local_chain_idx++;

        /* Count seeds in this chain by following confirmed successor links. */
        int chain_n = 0;
        int cur = head;
        while (cur < n_seeds) {
            chain_n++;
            int nxt = S_succ[cur];
            /* A link cur→nxt is confirmed only if S_preceding[nxt] == cur. */
            if (nxt >= n_seeds || (int)S_preceding[nxt] != cur) break;
            cur = nxt;
        }

        /* Allocate and fill seed array from this thread's pool shard. */
        mem_seed_t *chain_seeds_ptr = (mem_seed_t*)CUDAKernelMalloc(
                d_buffer_ptr, chain_n * sizeof(mem_seed_t), 8);
        int k = 0;
        cur = head;
        while (k < chain_n) {
            chain_seeds_ptr[k++] = seed_a[cur];
            int nxt = S_succ[cur];
            if (nxt >= n_seeds || (int)S_preceding[nxt] != cur) break;
            cur = nxt;
        }

        chain_a[chain_a_idx].pos      = seed_a[head].rbeg;
        chain_a[chain_a_idx].rid      = seed_a[head].rid;
        chain_a[chain_a_idx].is_alt   = !!d_bns->anns[seed_a[head].rid].is_alt;
        chain_a[chain_a_idx].frac_rep = seed_a[head].frac_rep;
        chain_a[chain_a_idx].n        = k;
        chain_a[chain_a_idx].m        = k;
        chain_a[chain_a_idx].seeds    = chain_seeds_ptr;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        d_chains[seqID].n = s_n_chains;
        d_chains[seqID].a = chain_a;
    }
}


/* sort chains of each read by weight
   shared-mem is pre-allocated to 3072*int
   assume that max(n_chn) is 3072
 */
__global__ void sort_chains_by_weight(mem_chain_v* d_chains, void* d_buffer_pools){
    int n_chn = d_chains[blockIdx.x].n;
    if (n_chn==0 || n_chn > 3072) return;
    void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, blockIdx.x%32);
    mem_chain_t* a = d_chains[blockIdx.x].a;	// array of chains

    extern __shared__ int SM[];			// shared mem, pre-allocated
    mem_chain_t** new_a_SM = (mem_chain_t**)SM;	// new array of chains on global mem
    uint16_t* w = (uint16_t*)&new_a_SM[1];		// array of weights
    uint16_t* new_i = (uint16_t*)&w[MAX_N_CHAIN]; // array of sorted chain index

    // calculate weight of each chain
    int n_iter = MAX_N_CHAIN/SORTCHAIN_BLOCKDIMX;
    for (int k=0; k<n_iter; k++){
        int i = k*blockDim.x + threadIdx.x;	// chainID to work on
        if (i<n_chn)
            w[i] = mem_chain_weight(&a[i]);
        else
            w[i] = 0;
    }
    __syncthreads(); __syncwarp();
    uint32_t thread_keys[NKEYS_EACH_THREAD];
    int thread_values[NKEYS_EACH_THREAD];
    for (int k=0; k<NKEYS_EACH_THREAD; k++){
        thread_values[k] = threadIdx.x*NKEYS_EACH_THREAD+k;
        thread_keys[k] = w[threadIdx.x*NKEYS_EACH_THREAD+k];
    }
    __syncthreads(); __syncwarp();
    // sort by composite key descending
    typedef cub::BlockRadixSort<uint32_t, SORTCHAIN_BLOCKDIMX, NKEYS_EACH_THREAD, int> BlockRadixSort;
    BlockRadixSort().SortDescending(thread_keys, thread_values);
    // transfer sorted index array (thread_values) to shared mem
    for (int k=0; k<NKEYS_EACH_THREAD; k++){
        new_i[threadIdx.x*NKEYS_EACH_THREAD+k] = thread_values[k];
    }

    // export output
    if (threadIdx.x==0){
        *new_a_SM = (mem_chain_t*)CUDAKernelMalloc(d_buffer_ptr, n_chn*sizeof(mem_chain_t), 8);
        d_chains[blockIdx.x].a = *new_a_SM;
    }
    __syncthreads(); __syncwarp();
    mem_chain_t* new_a = *new_a_SM;
    for (int k=0; k<n_iter; k++){
        int i = k*blockDim.x + threadIdx.x;	// chainID to work on
        if (i<n_chn){
            new_a[i] = a[new_i[i]];
            new_a[i].w = w[new_i[i]];
        }
    }
}


/* each block takes care of 1 read, do pairwise comparison of chains
   max number of chain is MAX_N_CHAIN
Notations (matching bwa-mem2 mem_chain_flt):
kept=0: dropped
kept=1: first-shadowed (retained for mapq accuracy)
kept=2: kept despite overlap (large_ovlp but not dropped)
kept=3: kept (no significant overlap with any higher-weight chain)
 */
__global__ void filter_chains(
        const mem_opt_t *opt,
        mem_chain_v *d_chains, 	// input and output
        void* d_buffer_pools)
{
    int i, j, n_chn, n_iter;
    n_chn = d_chains[blockIdx.x].n;
    mem_chain_t* a = d_chains[blockIdx.x].a;	// chains vector
    if (n_chn == 0) return; // no need to filter
    if (n_chn>MAX_N_CHAIN){
        return;
    }

    extern __shared__ int SM[];		// dynamic shared mem
    uint16_t* chn_beg_SM = (uint16_t*)SM; 	// start of chains
    uint16_t* chn_end_SM = &chn_beg_SM[MAX_N_CHAIN];	// end of chains
    uint16_t* chn_w_SM = &chn_end_SM[MAX_N_CHAIN];		// weight of chains
    uint8_t* chn_info_SM = (uint8_t*)&chn_w_SM[MAX_N_CHAIN]; // chains' kept and alt information
    int* reduce_SM = (int*)&chn_info_SM[MAX_N_CHAIN];         // 8-int warp-leader scratch

    // load data in SM
    n_iter = ceil((float)n_chn/blockDim.x);
    for (int k=0; k<n_iter; k++){
        i = k*blockDim.x+threadIdx.x; // chainID to work on
        if (i<n_chn){
            chn_beg_SM[i] = chn_beg(a[i]);
            chn_end_SM[i] = chn_end(a[i]);
            chn_w_SM[i] = a[i].w;
            chn_info_SM[i] = 0;
            if (i!=0) SET_KEPT(i,1);	// kept = 1 (pending)
            else SET_KEPT(i,3);			// chain 0 always kept
            SET_IS_ALT(i, a[i].is_alt);
            a[i].first = -1;
        }
    }
    __syncthreads(); __syncwarp();

    // Block-cooperative pairwise filter.
    // Outer i is block-serial (preserves bwa-mem2 ordering).
    // Thread t owns j in {t, t+256, t+512, ...} ∩ [0, i) — no atomics needed on a[j].first.
    // Block-min reduction honours break-on-drop; block-OR reduction tracks large_ovlp.
    {
        const int tid     = threadIdx.x;
        const int warpid  = tid >> 5;
        const int lane    = tid & 31;
        const int n_warps = blockDim.x >> 5;   // 8 warps for CHAIN_FLT_BLOCKSIZE=256

        const float mask_level  = opt->mask_level;
        const int   max_chn_gap = opt->max_chain_gap;
        const float drop_ratio  = opt->drop_ratio;
        const int   min_sl2     = opt->min_seed_len << 1;

        for (i = 1; i < n_chn; i++) {
            // Cache current chain's SM fields to avoid repeated broadcast reads
            const int ci_beg = chn_beg_SM[i];
            const int ci_end = chn_end_SM[i];
            const int ci_w   = chn_w_SM[i];
            const int ci_alt = GET_IS_ALT(i);
            const int ci_len = ci_end - ci_beg;

            // Phase 1: strided overlap scan
            uint32_t my_sig_bits = 0u;
            int local_drop_min   = i;    // sentinel: no drop found

            for (int k = 0; ; k++) {
                j = tid + k * (int)blockDim.x;
                if (j >= i) break;
                if (GET_KEPT(j) == 0) continue;
                int b_max = chn_beg_SM[j] > ci_beg ? chn_beg_SM[j] : ci_beg;
                int e_min = chn_end_SM[j] < ci_end ? chn_end_SM[j] : ci_end;
                if (e_min > b_max && (!GET_IS_ALT(j) || ci_alt)) {
                    int lj    = chn_end_SM[j] - chn_beg_SM[j];
                    int min_l = ci_len < lj ? ci_len : lj;
                    if (e_min - b_max >= min_l * mask_level && min_l < max_chn_gap) {
                        my_sig_bits |= (1u << k);
                        if (ci_w < chn_w_SM[j] * drop_ratio &&
                            chn_w_SM[j] - ci_w >= min_sl2)
                            local_drop_min = min(local_drop_min, j);
                    }
                }
            }
            __syncthreads();

            // Phase 2: block-min on local_drop_min -> jdrop (smallest j that would drop i)
            int wv = local_drop_min;
            for (int m = 16; m > 0; m >>= 1) wv = min(wv, __shfl_xor_sync(0xffffffff, wv, m));
            if (lane == 0) reduce_SM[warpid] = wv;
            __syncthreads();
            if (warpid == 0) {
                wv = (lane < n_warps) ? reduce_SM[lane] : i;
                for (int m = 4; m > 0; m >>= 1) wv = min(wv, __shfl_xor_sync(0xffffffff, wv, m));
                if (lane == 0) reduce_SM[0] = wv;
            }
            __syncthreads();
            const int jdrop = reduce_SM[0];

            // Phase 3: strided writeback for j in [0, jdrop] with significant overlap
            int local_sig_or = 0;
            for (int k = 0; ; k++) {
                j = tid + k * (int)blockDim.x;
                if (j >= i)    break;
                if (j > jdrop) break;
                if (my_sig_bits & (1u << k)) {
                    if (a[j].first < 0) a[j].first = i;
                    local_sig_or = 1;
                }
            }
            __syncthreads();

            // Phase 4: block-OR on local_sig_or -> has_sig (large_ovlp equivalent)
            wv = local_sig_or;
            for (int m = 16; m > 0; m >>= 1) wv |= __shfl_xor_sync(0xffffffff, wv, m);
            if (lane == 0) reduce_SM[warpid] = wv;
            __syncthreads();
            if (warpid == 0) {
                wv = (lane < n_warps) ? reduce_SM[lane] : 0;
                for (int m = 4; m > 0; m >>= 1) wv |= __shfl_xor_sync(0xffffffff, wv, m);
                if (lane == 0) reduce_SM[0] = wv;
            }
            __syncthreads();
            const int has_sig = reduce_SM[0];

            // Phase 5: thread 0 writes kept[i] (mirrors serial: drop=0, ovlp=2, none=3)
            if (tid == 0) {
                if (jdrop < i)
                    SET_KEPT(i, 0);
                else
                    SET_KEPT(i, has_sig ? 2 : 3);
            }
            __syncthreads();
        }
    }

    // Promote first-shadowed chains (matches bwa-mem2 post-loop).
    // a[j].first was set inline during pairwise comparison above.
    // For each chain with first >= 0, promote a[first] to kept=1.
    if (threadIdx.x==0){
        for (j=0; j<n_chn; j++){
            if (a[j].first >= 0) SET_KEPT(a[j].first, 1);
        }
    }
    __syncthreads(); __syncwarp();

    // do accounting of which chain is kept (kept >= 1)
    uint16_t* new_n_chn = chn_w_SM;		// chn_w_SM  now hold new n_chn
    uint16_t* old_index = chn_beg_SM;	// chn_beg_SM now hold index to the old chain
    mem_chain_t** new_a_SM = (mem_chain_t**)chn_end_SM;	// chn_end_SM now hold pointer to new_a
    if (threadIdx.x==0){
        new_n_chn[0] = 0;
        for (j=0; j<n_chn; j++){
            if (GET_KEPT(j)>=1){
                old_index[new_n_chn[0]++] = j;
            }
        }
        void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, blockIdx.x%32);
        *new_a_SM = (mem_chain_t*)CUDAKernelMalloc(d_buffer_ptr, new_n_chn[0]*sizeof(mem_chain_t), 8);
    }
    __syncthreads(); __syncwarp();
    // save to global data
    n_chn = new_n_chn[0];
    mem_chain_t* new_a = *new_a_SM;
    n_iter = ceil((float)n_chn/blockDim.x);
    for (int k=0; k<n_iter; k++){
        j = k*blockDim.x+threadIdx.x; // chainID to work on
        if (j<n_chn)
            new_a[j] = a[old_index[j]];
    }
    if (threadIdx.x==0){
        d_chains[blockIdx.x].n = n_chn;
        d_chains[blockIdx.x].a = new_a;
    }
}


/* filter_chained_seeds — block-per-read variant.
 *
 * Launch: <<<batch_size, FCS_BLOCKDIM>>> (one block per read).
 *
 * Within each block, FCS_BLOCKDIM threads cooperate to score seeds in
 * parallel.  Thread t handles seeds t, t+blockDim.x, t+2*blockDim.x, …
 * across all chains.  Thread 0 performs the serial compaction steps.
 *
 * For typical 150 bp reads the early-exit condition (min_l >
 * MEM_SEEDSW_COEF * l_query) fires, so no SW work is done and the
 * kernel is nearly free.  The parallel design benefits longer reads
 * (≥ 700 bp) where the early exit does not fire.
 */
__global__ void filter_chained_seeds(
        const mem_opt_t *d_opt, const bntseq_t *d_bns, const uint8_t *d_pac,
        const uint8_t *d_seq, const int *d_seq_offset,
        mem_chain_v *d_chains,
        int n,
        void* d_buffer_pools
        )
{
    int read_id = blockIdx.x;
    if (read_id >= n) return;

    __shared__ int          s_n_chn;
    __shared__ mem_chain_t *s_a;
    __shared__ int          s_l_query;
    __shared__ const uint8_t *s_query;
    __shared__ int          s_min_HSP_score;
    __shared__ int          s_skip;

    if (threadIdx.x == 0) {
        int seq_off  = d_seq_offset[read_id];
        s_n_chn      = d_chains[read_id].n;
        s_a          = d_chains[read_id].a;
        s_l_query    = d_seq_offset[read_id + 1] - seq_off;
        s_query      = &d_seq[seq_off];
        double min_l = d_opt->min_chain_weight
            ? (double)(MEM_HSP_COEF * d_opt->min_chain_weight)
            : MEM_MINSC_COEF * log((double)s_l_query);
        s_min_HSP_score = (int)(d_opt->a * min_l + .499);
        s_skip = (s_n_chn == 0 || min_l > MEM_SEEDSW_COEF * (double)s_l_query) ? 1 : 0;
    }
    __syncthreads();

    if (s_skip) return;

    /* Parallel seed scoring: each thread owns a stripe of seeds per chain. */
    for (int ci = 0; ci < s_n_chn; ci++) {
        mem_chain_t *c = &s_a[ci];
        int n_seeds = c->n;

        /* Spread threads across pools 0-31 to reduce atomicAdd contention. */
        void *d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools,
            (blockIdx.x * blockDim.x + threadIdx.x) % 32);

        for (int si = threadIdx.x; si < n_seeds; si += blockDim.x) {
            mem_seed_t *s = &c->seeds[si];
            s->score = mem_seed_sw(d_opt, d_bns, d_pac,
                                   s_l_query, s_query, s, d_buffer_ptr);
        }
        __syncthreads();

        /* Thread 0: compact seeds that failed the score threshold. */
        if (threadIdx.x == 0) {
            int k = 0;
            for (int j = 0; j < n_seeds; j++) {
                mem_seed_t *s = &c->seeds[j];
                if (s->score < 0 || s->score >= s_min_HSP_score) {
                    s->score = s->score < 0 ? s->len * d_opt->a : s->score;
                    c->seeds[k++] = *s;
                }
            }
            c->n = k;
        }
        __syncthreads();
    }

    /* Thread 0: compact chains with zero seeds remaining. */
    if (threadIdx.x == 0) {
        int k = 0;
        for (int ci = 0; ci < s_n_chn; ci++) {
            if (s_a[ci].n > 0) {
                if (k != ci) s_a[k] = s_a[ci];
                k++;
            }
        }
        d_chains[read_id].n = k;
    }
}
