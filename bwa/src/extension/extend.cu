/*
 * extend.cu -- SW extension kernels: sw_seed_prep, extend_and_sw (fused pair-gen + SW extension).
 */

/*
 * This file is BWA-MEM2-MIT-derived (e.g. cal_max_gap, seed extension prep
 * logic, ported from bwa-mem2's CPU code), but the kernels here call into
 * GPL-3.0-derived code from minhhpham/bwa (https://github.com/minhhpham/bwa,
 * Copyright (c) minhhpham) — specifically ksw_extend_warp2 in
 * extension/ksw.cu, a GPU port of BWA-MEM (Copyright (c) Dana-Farber Cancer
 * Institute, Broad Institute, Genome Research Ltd.). Because of that
 * dependency this file is part of G3SA and is licensed under GPL-3.0 (see
 * LICENSE).
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

#include "extend.cuh"

// 2-bit packed reference decoding macro (also defined in bntseq.cu)
#define _get_pac(pac, l) ((pac)[(l)>>2]>>((~(l)&3)<<1)&3)


/****************************************
 * Construct the alignment from a chain *
 ****************************************/

__device__ static inline int cal_max_gap(const mem_opt_t *opt, int qlen)
{
    int l_del = (int)((double)(qlen * opt->a - opt->o_del) / opt->e_del + 1.);
    int l_ins = (int)((double)(qlen * opt->a - opt->o_ins) / opt->e_ins + 1.);
    int l = l_del > l_ins? l_del : l_ins;
    l = l > 1? l : 1;
    return l < opt->w<<1? l : opt->w<<1;
}


/* preprocessing 1 for SW extension
   count the number of seeds for each read and write to global records, allocate output regs vector
 */

// Each CUDA thread sums up the number of seeds in all chains
// of each read sequence. Same seed could be counted multiple times
// as it can be contained in multiple chains. These sums of seeds per
// read are atomically sumed up in *d_Nseeds.
//
// Then, concatenate all seeds in all chains in each read (= SWseeds)
// (including duplicates) into a preallocated 1D array in the
// global memory region (d_seed_records).
//
// A mem_alnreg_v vector is allocated to contain all SWseed extensions
// for each read.
__global__ void sw_seed_prep(
        mem_chain_v *d_chains,
        mem_alnreg_v *d_regs,
        seed_record_t *d_seed_records,
        int *d_Nseeds,	// total seed count across all reads
        int n_seqs,	// number of reads
        void* d_buffer_pools
        )
{
    // CUB BlockScan to reduce per-thread atomicAdd to one per block
    typedef cub::BlockScan<int, 32> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ int block_base;  // base offset from global atomicAdd

    int seqID = blockIdx.x*blockDim.x+threadIdx.x;	// ID of the read to process
    int chn_n = 0;
    mem_chain_t* chn_a = NULL;

    if (seqID < n_seqs) {
        chn_n = d_chains[seqID].n;					// n_chains of this read
        chn_a = d_chains[seqID].a;			// chain array of this read
    }

    // count number of seeds for this read
    int n_seeds = 0;
    for (int i=0; i<chn_n; i++)	// loop through chains
        n_seeds = n_seeds + chn_a[i].n;

    // BlockScan: exclusive prefix sum gives each thread its offset within the block
    int thread_prefix;
    int block_total;
    BlockScan(temp_storage).ExclusiveSum(n_seeds, thread_prefix, block_total);

    // Thread 0 does a single global atomicAdd for the whole block
    if (threadIdx.x == 0)
        block_base = atomicAdd(d_Nseeds, block_total);
    __syncthreads();

    // Each thread computes its personal start offset
    int start = block_base + thread_prefix;

    // Early exit for out-of-bounds or empty reads
    if (seqID >= n_seqs) return;

    if (chn_n == 0 || n_seeds == 0){
        d_regs[seqID].n = 0;
        return;
    }
    void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, 32 + threadIdx.x%32);

    // write seed record to global d_seed_records
    int j = 0;	// start+j will be the offset on d_seed_records, j is regID
    for (int i=0; i<chn_n; i++){	// i is chainID
        if (chn_a[i].n==0) continue;
        // start+j == batch-level
        for (int k=0; k<chn_a[i].n; k++){
            d_seed_records[start+j].seqID = seqID; //input idx
            d_seed_records[start+j].chainID = (uint16_t)i;// chain idx
            d_seed_records[start+j].seedID = (uint16_t)k; //chain-level
            d_seed_records[start+j].regID = (uint16_t)j; //input-level
            j++;
        }
    }

    // allocate regs vector
    d_regs[seqID].n = d_regs[seqID].m = n_seeds;
    d_regs[seqID].a = (mem_alnreg_t*)CUDAKernelCalloc(d_buffer_ptr, n_seeds, sizeof(mem_alnreg_t), 8);
}

/* Fused pair-generation + SW extension.
 *
 * Eliminates the 80 B/seed global-memory round-trip (40 B write + 40 B read)
 * through d_seed_records that a separate pair-gen kernel + extension kernel
 * would otherwise require.  The pointer/length pairs computed in the first
 * half (qs_left/rs_left/qs_right/rs_right and their lengths) stay in
 * registers and are consumed directly by the SW extension second half.
 *
 * d_seed_records is read for seed metadata (seqID/chainID/seedID/regID) but
 * NOT written by this kernel.
 *
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
        seed_record_t *d_seed_records,   // read metadata fields only (seqID/chainID/seedID/regID)
        int *d_Nseeds,
        int n_seqs,
        void* d_buffer_pools
        )
{
    if (blockDim.x != 32) { printf("wrong blocksize config\n"); __trap(); }
    int i = blockIdx.x;
    if (i >= d_Nseeds[0]) return;
    int lane = threadIdx.x;

    // -----------------------------------------------------------------------
    // Phase 1: extend_pair_generate body — compute buffers, keep in registers
    // -----------------------------------------------------------------------

    // Thread 0 reads metadata; broadcast to all lanes
    int seqID, chainID, seedID, regID;
    int l_seq;
    uint8_t *seq;
    int64_t rbeg_val;
    int32_t qbeg_val, slen_val;

    if (lane == 0) {
        seqID   = d_seed_records[i].seqID;
        chainID = (int)d_seed_records[i].chainID;
        seedID  = (int)d_seed_records[i].seedID;
        regID   = (int)d_seed_records[i].regID;
        int seq_offset      = d_seq_offset[seqID];
        int seq_offset_next = d_seq_offset[seqID + 1];
        l_seq = seq_offset_next - seq_offset;
        seq   = d_seq + seq_offset;
        mem_seed_t *seed = &(d_chains[seqID].a[chainID].seeds[seedID]);
        rbeg_val = seed->rbeg;
        qbeg_val = seed->qbeg;
        slen_val = seed->len;
    }
    seqID     = __shfl_sync(0xFFFFFFFF, seqID, 0);
    chainID   = __shfl_sync(0xFFFFFFFF, chainID, 0);
    seedID    = __shfl_sync(0xFFFFFFFF, seedID, 0);
    regID     = __shfl_sync(0xFFFFFFFF, regID, 0);
    l_seq     = __shfl_sync(0xFFFFFFFF, l_seq, 0);
    seq       = (uint8_t*)(unsigned long long)__shfl_sync(0xFFFFFFFF, (unsigned long long)seq, 0);
    rbeg_val  = (int64_t)__shfl_sync(0xFFFFFFFF, (unsigned long long)rbeg_val, 0);
    qbeg_val  = __shfl_sync(0xFFFFFFFF, qbeg_val, 0);
    slen_val  = __shfl_sync(0xFFFFFFFF, slen_val, 0);

    // Warp-cooperative rmax computation
    mem_chain_t *chain = &(d_chains[seqID].a[chainID]);
    int chain_n = chain->n;
    int64_t l_pac = d_bns->l_pac;

    int64_t my_rmax0 = l_pac << 1;
    int64_t my_rmax1 = 0;
    for (int k = lane; k < chain_n; k += 32) {
        int64_t r  = chain->seeds[k].rbeg;
        int32_t q  = chain->seeds[k].qbeg;
        int32_t ln = chain->seeds[k].len;
        int64_t b = r - (q + cal_max_gap(d_opt, q));
        int64_t e = r + ln + ((l_seq - q - ln) + cal_max_gap(d_opt, l_seq - q - ln));
        my_rmax0 = my_rmax0 < b ? my_rmax0 : b;
        my_rmax1 = my_rmax1 > e ? my_rmax1 : e;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        int64_t other0 = (int64_t)__shfl_xor_sync(0xFFFFFFFF, (unsigned long long)my_rmax0, offset);
        int64_t other1 = (int64_t)__shfl_xor_sync(0xFFFFFFFF, (unsigned long long)my_rmax1, offset);
        my_rmax0 = my_rmax0 < other0 ? my_rmax0 : other0;
        my_rmax1 = my_rmax1 > other1 ? my_rmax1 : other1;
    }
    int64_t rmax0_val = my_rmax0 > 0 ? my_rmax0 : 0;
    int64_t rmax1_val = my_rmax1 < (l_pac << 1) ? my_rmax1 : (l_pac << 1);
    if (rmax0_val < l_pac && l_pac < rmax1_val) {
        if (chain->seeds[0].rbeg < l_pac) rmax1_val = l_pac;
        else rmax0_val = l_pac;
    }

    // Thread 0: binary search for rid
    int rid;
    int64_t ref_beg = rmax0_val, ref_end = rmax1_val;
    int64_t ref_len;
    int is_rev_strand;

    if (lane == 0) {
        if (ref_end < ref_beg) { int64_t t = ref_end; ref_end = ref_beg; ref_beg = t; }
        int is_rev;
        int64_t depos = (is_rev = (chain->seeds[0].rbeg >= l_pac)) ? (l_pac<<1) - 1 - chain->seeds[0].rbeg : chain->seeds[0].rbeg;
        rid = bns_pos2rid_gpu(d_bns, depos);
        int64_t far_beg = d_bns->anns[rid].offset;
        int64_t far_end = far_beg + d_bns->anns[rid].len;
        if (is_rev) {
            int64_t tmp = far_beg;
            far_beg = (l_pac << 1) - far_end;
            far_end = (l_pac << 1) - tmp;
        }
        ref_beg = ref_beg > far_beg ? ref_beg : far_beg;
        ref_end = ref_end < far_end ? ref_end : far_end;
        is_rev_strand = (ref_beg >= l_pac) ? 1 : 0;
        d_regs[seqID].a[regID].rid = rid;
    }
    ref_beg       = (int64_t)__shfl_sync(0xFFFFFFFF, (unsigned long long)ref_beg, 0);
    ref_end       = (int64_t)__shfl_sync(0xFFFFFFFF, (unsigned long long)ref_end, 0);
    is_rev_strand = __shfl_sync(0xFFFFFFFF, is_rev_strand, 0);
    ref_len = ref_end - ref_beg;

    // Thread 0 allocates reference decode buffer
    uint8_t *rseq = 0;
    if (ref_len > 0) {
        if (lane == 0) {
            void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, 32 + blockIdx.x % 32);
            rseq = (uint8_t*)CUDAKernelMalloc(d_buffer_ptr, ref_len, 1);
        }
        rseq = (uint8_t*)(unsigned long long)__shfl_sync(0xFFFFFFFF, (unsigned long long)rseq, 0);

        // Warp-cooperative PAC decode
        if (is_rev_strand) {
            int64_t end_f = (l_pac << 1) - 1 - ref_beg;
            for (int64_t idx = lane; idx < ref_len; idx += 32) {
                int64_t k = end_f - idx;
                rseq[idx] = 3 - _get_pac(d_pac, k);
            }
        } else {
            for (int64_t idx = lane; idx < ref_len; idx += 32) {
                rseq[idx] = _get_pac(d_pac, ref_beg + idx);
            }
        }
    }

    rmax0_val = ref_beg;
    rmax1_val = ref_end;

    // Left extension: reversed copies with warp-cooperative memcpy
    int qlen_left = qbeg_val;
    int rlen_left = (int)(rbeg_val - rmax0_val);
    uint8_t *qs_left = 0, *rs_left = 0;

    if (qlen_left > 0) {
        uint8_t *buf = 0;
        if (lane == 0) {
            void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, 32 + blockIdx.x % 32);
            buf = (uint8_t*)CUDAKernelMalloc(d_buffer_ptr, qlen_left + rlen_left, 1);
        }
        buf = (uint8_t*)(unsigned long long)__shfl_sync(0xFFFFFFFF, (unsigned long long)buf, 0);
        qs_left = buf;
        rs_left = buf + qlen_left;

        for (int r = lane; r < qlen_left; r += 32)
            qs_left[r] = seq[qlen_left - 1 - r];
        for (int r = lane; r < rlen_left; r += 32)
            rs_left[r] = rseq[rlen_left - 1 - r];
    }

    // Right extension: pointer arithmetic only, no copy
    int qlen_right = l_seq - qbeg_val - slen_val;
    int rlen_right = (int)(rmax1_val - rbeg_val - slen_val);
    uint8_t *qs_right = (qlen_right > 0) ? seq + qbeg_val + slen_val : 0;
    uint8_t *rs_right = (qlen_right > 0) ? rseq + (int)(rbeg_val - rmax0_val) + slen_val : 0;

    // -----------------------------------------------------------------------
    // Phase 2: local_extend body — SW extension using register-held buffers
    // -----------------------------------------------------------------------

    int score;
    int gscore;
    int qle, tle;
    int gtle;
    int query_end;
    int64_t ref_end_sw;

    int w = d_opt->w;
    int end_bonus = d_opt->pen_clip5;

    // left extension
    int h0 = chain->seeds[seedID].len * d_opt->a;
    uint8_t *query = qs_left;
    int qlen  = qlen_left;
    uint8_t *target = rs_left;
    int tlen  = rlen_left;
    int w_left = w, w_right = w;

    if (qlen > 0) {
        int max_off = 0;
        score = ksw_extend_warp2(qlen, query, tlen, target, 5, d_opt->mat, d_opt->o_del, d_opt->e_del, d_opt->o_ins, d_opt->e_ins, w_left, end_bonus, h0, &qle, &tle, &gtle, &gscore, &max_off);
        score  = __shfl_sync(0xffffffff, score,  0);
        qle    = __shfl_sync(0xffffffff, qle,    0);
        tle    = __shfl_sync(0xffffffff, tle,    0);
        gtle   = __shfl_sync(0xffffffff, gtle,   0);
        gscore = __shfl_sync(0xffffffff, gscore, 0);
        max_off= __shfl_sync(0xffffffff, max_off,0);
        if (score > 0 && max_off >= (w_left >> 1) + (w_left >> 2)) {
            w_left <<= 1;
            score = ksw_extend_warp2(qlen, query, tlen, target, 5, d_opt->mat, d_opt->o_del, d_opt->e_del, d_opt->o_ins, d_opt->e_ins, w_left, end_bonus, h0, &qle, &tle, &gtle, &gscore, &max_off);
            score  = __shfl_sync(0xffffffff, score,  0);
            qle    = __shfl_sync(0xffffffff, qle,    0);
            tle    = __shfl_sync(0xffffffff, tle,    0);
            gtle   = __shfl_sync(0xffffffff, gtle,   0);
            gscore = __shfl_sync(0xffffffff, gscore, 0);
        }
        if (gscore <= 0 || gscore <= (score - d_opt->pen_clip5)) {
            query_end = chain->seeds[seedID].qbeg - qle;
            ref_end_sw = chain->seeds[seedID].rbeg - tle;
        } else {
            query_end = 0;
            ref_end_sw = chain->seeds[seedID].rbeg - gtle;
        }
    } else { score = h0; query_end = 0; ref_end_sw = chain->seeds[seedID].rbeg; }

    int truesc;
    if (qlen > 0) {
        if (gscore <= 0 || gscore <= score - d_opt->pen_clip5)
            truesc = score;
        else
            truesc = gscore;
    } else {
        truesc = score;
    }
    int left_qb = query_end;
    int64_t left_rb = ref_end_sw;
    if (threadIdx.x == 0) {
        d_regs[seqID].a[regID].score = score;
        d_regs[seqID].a[regID].qb = query_end;
        d_regs[seqID].a[regID].rb = ref_end_sw;
    }

    // right extension
    h0 = score;
    query  = qs_right;
    qlen   = qlen_right;
    target = rs_right;
    tlen   = rlen_right;
    if (qlen > 0) {
        int max_off = 0;
        end_bonus = d_opt->pen_clip3;
        score = ksw_extend_warp2(qlen, query, tlen, target, 5, d_opt->mat, d_opt->o_del, d_opt->e_del, d_opt->o_ins, d_opt->e_ins, w_right, end_bonus, h0, &qle, &tle, &gtle, &gscore, &max_off);
        score  = __shfl_sync(0xffffffff, score,  0);
        qle    = __shfl_sync(0xffffffff, qle,    0);
        tle    = __shfl_sync(0xffffffff, tle,    0);
        gtle   = __shfl_sync(0xffffffff, gtle,   0);
        gscore = __shfl_sync(0xffffffff, gscore, 0);
        max_off= __shfl_sync(0xffffffff, max_off,0);
        if (score > 0 && max_off >= (w_right >> 1) + (w_right >> 2)) {
            w_right <<= 1;
            score = ksw_extend_warp2(qlen, query, tlen, target, 5, d_opt->mat, d_opt->o_del, d_opt->e_del, d_opt->o_ins, d_opt->e_ins, w_right, end_bonus, h0, &qle, &tle, &gtle, &gscore, &max_off);
            score  = __shfl_sync(0xffffffff, score,  0);
            qle    = __shfl_sync(0xffffffff, qle,    0);
            tle    = __shfl_sync(0xffffffff, tle,    0);
            gtle   = __shfl_sync(0xffffffff, gtle,   0);
            gscore = __shfl_sync(0xffffffff, gscore, 0);
        }
        if (gscore <= 0 || gscore <= (score - d_opt->pen_clip3)) {
            query_end = chain->seeds[seedID].qbeg + chain->seeds[seedID].len + qle;
            ref_end_sw = chain->seeds[seedID].rbeg + chain->seeds[seedID].len + tle;
        } else {
            query_end = chain->seeds[seedID].qbeg + chain->seeds[seedID].len + qlen;
            ref_end_sw = chain->seeds[seedID].rbeg + chain->seeds[seedID].len + gtle;
        }
    } else {
        score = h0;
        query_end = chain->seeds[seedID].qbeg + chain->seeds[seedID].len;
        ref_end_sw = chain->seeds[seedID].rbeg + chain->seeds[seedID].len;
    }
    if (qlen > 0) {
        if (gscore <= 0 || gscore <= score - d_opt->pen_clip3)
            truesc += score - h0;
        else
            truesc += gscore - h0;
    }

    int nseeds = chain->n;
    int seedcov = 0;
    for (int s = 0; s < nseeds; s++) {
        mem_seed_t *t = &(chain->seeds[s]);
        if (t->qbeg >= left_qb && t->qbeg + t->len <= query_end &&
            t->rbeg >= left_rb && t->rbeg + t->len <= ref_end_sw)
            seedcov += t->len;
    }
    if (threadIdx.x == 0) {
        d_regs[seqID].a[regID].score    = score;
        d_regs[seqID].a[regID].qe       = query_end;
        d_regs[seqID].a[regID].re       = ref_end_sw;
        d_regs[seqID].a[regID].w        = w_left > w_right ? w_left : w_right;
        d_regs[seqID].a[regID].seedlen0 = chain->seeds[seedID].len;
        d_regs[seqID].a[regID].frac_rep = chain->frac_rep;
        d_regs[seqID].a[regID].seedcov  = seedcov;
        d_regs[seqID].a[regID].truesc   = truesc;
        d_regs[seqID].a[regID].hash     = (uint64_t)chain->seeds[seedID].rbeg;
        d_regs[seqID].a[regID].alt_sc   = chain->seeds[seedID].qbeg;
    }
}

