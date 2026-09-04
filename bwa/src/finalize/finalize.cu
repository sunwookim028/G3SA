/*
 * finalize.cu -- Traceback preprocessing, fused traceback + finalize kernel.
 *                Also contains mem_approx_mapq_se (MAPQ computation).
 */

/*
 * Portions of this file are adapted from minhhpham/bwa
 * (https://github.com/minhhpham/bwa), Copyright (c) minhhpham,
 * GPL-3.0 licensed — a GPU port of BWA-MEM (Copyright (c) Dana-Farber
 * Cancer Institute, Broad Institute, Genome Research Ltd.). This file is
 * part of G3SA and is licensed under GPL-3.0 (see LICENSE).
 *
 * bns_depos_gpu, infer_bw, and mem_approx_mapq_se below are near-byte-identical
 * to minhhpham/bwa's bwamem_GPU.cu. The traceback/MAPQ orchestration kernels
 * (finalize_prep1, finalize_prep2, traceback_and_finalize) are this project's
 * own work.
 */

#include "gpu_types.h"
#include "gmem_alloc.cuh"
#include "bntseq.h"
#include "ksw.cuh"
#include "kstring_device.cuh"
#include <string.h>
#include "cuda_wrapper.h"
#include "macro.h"

#include "finalize.cuh"


/*****************************************************
 * Device functions for generating alignment results *
 *****************************************************/

__device__ static inline int64_t bns_depos_gpu(const bntseq_t *bns, int64_t pos, int *is_rev)
{
    return (*is_rev = (pos >= bns->l_pac))? (bns->l_pac<<1) - 1 - pos : pos;
}

__device__ static int mem_approx_mapq_se(const mem_opt_t *opt, const mem_alnreg_t *a)
{
    int mapq, l, sub = a->sub? a->sub : opt->min_seed_len * opt->a;
    double identity;
    sub = a->csub > sub? a->csub : sub;
    if (sub >= a->score) return 0;
    l = a->qe - a->qb > a->re - a->rb? a->qe - a->qb : a->re - a->rb;
    identity = 1. - (double)(l * opt->a - a->score) / (opt->a + opt->b) / l;
    if (a->score == 0) {
        mapq = 0;
    } else if (opt->mapQ_coef_len > 0) {
        double tmp;
        tmp = l < opt->mapQ_coef_len? 1. : opt->mapQ_coef_fac / log((double)l);
        tmp *= identity * identity;
        mapq = (int)(6.02 * (a->score - sub) / opt->a * tmp * tmp + .499);
    } else {
        mapq = (int)(MEM_MAPQ_COEF * (1. - (double)sub / a->score) * log((double)a->seedcov) + .499);
        mapq = identity < 0.95? (int)(mapq * identity * identity + .499) : mapq;
    }
    if (a->sub_n > 0) mapq -= (int)(4.343 * log((double)a->sub_n+1) + .499);
    if (mapq > 60) mapq = 60;
    if (mapq < 0) mapq = 0;
    mapq = (int)(mapq * (1. - a->frac_rep) + .499);
    return mapq;
}

__device__ static inline int infer_bw(int l1, int l2, int score, int a, int q, int r)
{
    int w;
    if (l1 == l2 && l1 * a - score < (q + r - a)<<1) return 0; // to get equal alignment length, we need at least two gaps
    w = ((double)((l1 < l2? l1 : l2) * a - score - q) / r + 2.);
    if (w < abs(l1 - l2)) w = abs(l1 - l2);
    return w;
}


/* prepare ref sequence for global SW
   allocate mem_aln_v array for each read
   write to d_seed_records just like SW extension,
   - seqID
   - regID: index on d_regs and d_alns
   if read has no good alignment, write an unmapped record:
   - rid = -1
   - pos = -1
   - flag = 0x4
 */
__global__ void finalize_prep1(
        int batch_size,
        mem_alnreg_v* d_regs,
        mem_aln_v * d_alns,
        seed_record_t *d_seed_records,
        int *d_Nseeds,
        void* d_buffer_pools)
{
    int seqID = blockIdx.x*blockDim.x + threadIdx.x;
    if(seqID >= batch_size) return;
    // allocate mem_aln_t array
    int n_aln = d_regs[seqID].n;

    // first create record on d_alns
    void *d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, 32 + threadIdx.x%32);
    d_alns[seqID].n = n_aln;
    // legit records if n_aln>0
    if (n_aln>0) d_alns[seqID].a = (mem_aln_t*)CUDAKernelCalloc(d_buffer_ptr, n_aln, sizeof(mem_aln_t), 8);
    // create unmapped records otherwise
    else {
        d_alns[seqID].a = (mem_aln_t*)CUDAKernelCalloc(d_buffer_ptr, 1, sizeof(mem_aln_t), 8);
        d_alns[seqID].a[0].rid = -1;
        d_alns[seqID].a[0].pos = -1;
        d_alns[seqID].a[0].flag = 0x4;
    }
    // atomic add n_seeds at block level
    __shared__ int S_block_nseeds[1];	// total seeds in this block
    __shared__ int S_block_offset[1];	// block's offset on d_seed_records
    if (threadIdx.x==0) S_block_nseeds[0] = 0;
    __syncthreads();
    int thread_offset;
    // create an unmapped record if no good aln
    if (n_aln<=0) n_aln = 1;
    thread_offset = atomicAdd(&S_block_nseeds[0], n_aln);
    __syncthreads();
    if (threadIdx.x==0) S_block_offset[0] = atomicAdd(d_Nseeds, S_block_nseeds[0]);
    __syncthreads();

    for (int i=0; i<n_aln; i++){
        int offset = S_block_offset[0] + thread_offset + i;
        d_seed_records[offset].seqID = seqID;
        d_seed_records[offset].regID = i;	// alnID
    }
}


/* run at aln level
   prepare seqs for SW global
   - .read_right: query
   - .readlen_right: lquery
   - .ref_right: reference
   - .reflen_right: lref
   - .readlen_left: bandwidth
   - .reflen_left: whether cigar should be reversed (1) or not (0)
   calculate bandwidth for SW global
   store l_ref*w to d_sortkeys_in and seqID to d_seqIDs_in
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
        void* d_buffer_pools)
{
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    if (offset>=Nseeds) return;
    int seqID = d_seed_records[offset].seqID;
    int alnID = d_seed_records[offset].regID;
    if (d_alns[seqID].a[alnID].rid == -1) return; // ignore unmapped records

    void *d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, 32 + threadIdx.x%32);
    // prepare SW sequences
    int64_t rb = d_regs[seqID].a[alnID].rb;
    int64_t re = d_regs[seqID].a[alnID].re;
    int qb = d_regs[seqID].a[alnID].qb;
    int qe = d_regs[seqID].a[alnID].qe;
    uint8_t *query;
    int seq_offset = d_seq_offset[seqID];
    query = d_seq + seq_offset + qb;
    int l_query = qe - qb;
    int64_t rlen;
    int64_t l_pac = d_bns->l_pac;
    // Cap reference extraction to bound ksw_global3 pool usage.
    // The banded global SW (w <= GLOBALSW_BANDWITH_CUTOFF) cannot access
    // reference positions beyond l_query + GLOBALSW_BANDWITH_CUTOFF from rb,
    // so there is no accuracy loss from this cap (the bandwidth cap already
    // prevents correct resolution of deletions larger than GLOBALSW_BANDWITH_CUTOFF).
    {
        int64_t max_rlen = (int64_t)l_query + GLOBALSW_BANDWITH_CUTOFF + 16;
        if (re - rb > max_rlen) re = rb + max_rlen;
    }
    uint8_t *rseq = bns_get_seq_gpu(l_pac, d_pac, rb, re, &rlen, d_buffer_ptr);
    // calculate bandwidth
    int w;
    if (l_query == re-rb){ w=0; }	// no gap, no need to do DP
    else{
        int a = d_opt->a;
        int o_del = d_opt->o_del;
        int e_del = d_opt->e_del;
        int o_ins = d_opt->o_ins;
        int e_ins = d_opt->e_ins;
        int tmp;
        // inferred bandwidth
        w   = infer_bw(l_query, re-rb, d_regs[seqID].a[alnID].score, a, o_del, e_del);
        tmp = infer_bw(l_query, re-rb, d_regs[seqID].a[alnID].score, a, o_ins, e_ins);
        w = w>tmp? w : tmp;
        // global bandwidth
        int max_gap, max_ins, max_del;
        max_ins = (int)((double)(((l_query+1)>>1) * a - o_ins) / e_ins + 1.);
        max_del = (int)((double)(((l_query+1)>>1) * a - o_del) / e_del + 1.);
        max_gap = max_ins > max_del? max_ins : max_del;
        max_gap = max_gap > 1? max_gap : 1;
        tmp = (max_gap + abs((int)rlen - l_query) + 1) >> 1;
        w = w<tmp? w : tmp;
        tmp = abs((int)rlen - l_query) + 3;
        w = w>tmp? w : tmp;
    }
    // save these info to d_seed_records for next kernel
    d_seed_records[offset].read_right = query;		// query for SW
    d_seed_records[offset].readlen_right = l_query;
    d_seed_records[offset].ref_right = rseq;		// target for SW
    d_seed_records[offset].reflen_right = rlen;
    if (rb>=l_pac) d_seed_records[offset].reflen_left = 1; // signal that cigar need to be reversed
    else d_seed_records[offset].reflen_left = 0;
    d_seed_records[offset].readlen_left = (uint16_t)w;	// bandwidth
    d_sortkeys_in[offset] = w*rlen;		// for sorting
    d_seqIDs_in[offset] = offset;		// for sorting
}

/* Fused traceback + finalize: performs global SW, NM calculation, and all
   finalize work (pos/rid/cigar fixup, MAPQ, output packing) in a single kernel.

   Launch: <<<Nseeds, 32>>> — one warp (block) per alignment.
   Thread 0 runs ksw_global3 (serial global SW), then handles traceback, NM,
   finalize, and output packing. Threads 1–31 idle during global SW.
 */
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
        )
{
    // One block per alignment (warp-cooperative)
    int seedIdx = blockIdx.x;
    if (seedIdx >= Nseeds) return;
    int ID = d_seqIDs_out[seedIdx];  // map to new ID after sorting
    int seqID = d_seed_records[ID].seqID;
    int alnID = d_seed_records[ID].regID;
    if (d_alns[seqID].a[alnID].rid == -1) return; // ignore unmapped records

    // All threads in this block share one buffer pool (selected by block ID)
    void *d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, 32 + blockIdx.x % 32);

    // === TRACEBACK PART (all threads cooperate on DP) ===
    int bandwidth = (int)d_seed_records[ID].readlen_left;
    if (bandwidth>=GLOBALSW_BANDWITH_CUTOFF) bandwidth = GLOBALSW_BANDWITH_CUTOFF;
    uint8_t *query = d_seed_records[ID].read_right;
    int l_query_sw = (int)d_seed_records[ID].readlen_right;
    uint8_t *target = d_seed_records[ID].ref_right;
    int l_target = (int)d_seed_records[ID].reflen_right;
    // calculate cigar and score
    uint32_t *cigar = NULL; int n_cigar = 0, score = 0;
    if (bandwidth==0){
        // No gap — all threads idle except thread 0
        if (threadIdx.x == 0) {
            cigar = (uint32_t*)CUDAKernelMalloc(d_buffer_ptr, 4, 4);
            cigar[0] = l_query_sw<<4 | 0;
            n_cigar = 1;
            score = 0;
            for (int i = 0; i < l_query_sw; ++i)
                score += d_opt->mat[target[i]*5 + query[i]];
        }
    } else {
        if (threadIdx.x == 0) {
            // Fast path for short reads: ksw_global2 uses a full direction matrix
            // (no grid-blocked re-fill), halving DP work when the matrix fits in pool.
            // Product gate: n_col = min(l_query_sw, 2*bandwidth+1); allow ksw_global2
            // when n_col * l_target < 16384 (16 KB/alignment pool budget).  This fires
            // for ~90% of 76bp reads.
            if (l_query_sw < 100) {
                int n_col = l_query_sw < 2 * bandwidth + 1 ? l_query_sw : 2 * bandwidth + 1;
                if ((long)n_col * l_target < 16384) {
                    score = ksw_global2(l_query_sw, query, l_target, target, 5, d_opt->mat, d_opt->o_del, d_opt->e_del, d_opt->o_ins, d_opt->e_ins, bandwidth, &n_cigar, &cigar, d_buffer_ptr);
                } else {
                    score = ksw_global3(l_query_sw, query, l_target, target, 5, d_opt->mat, d_opt->o_del, d_opt->e_del, d_opt->o_ins, d_opt->e_ins, bandwidth, &n_cigar, &cigar, d_buffer_ptr);
                }
            } else {
                score = ksw_global3(l_query_sw, query, l_target, target, 5, d_opt->mat, d_opt->o_del, d_opt->e_del, d_opt->o_ins, d_opt->e_ins, bandwidth, &n_cigar, &cigar, d_buffer_ptr);
            }
        }
    }

    // === Everything below is thread 0 only ===
    if (threadIdx.x != 0) return;

    // calculate NM
    int NM;
    {
        int k, x, y, u, n_mm = 0, n_gap = 0;
        kstring_t str; str.l = str.m = n_cigar*4; str.s = (char*)cigar;
        const char *int2base = (d_seed_records[ID].reflen_left==0)? "ACGTN" : "TGCAN";
        for (k = 0, x = y = u = 0; k < n_cigar; ++k) {
            int op, len;
            cigar = (uint32_t*)str.s;
            op  = cigar[k]&0xf, len = cigar[k]>>4;
            if (op == 0) { // match
                for (int i = 0; i < len; ++i) {
                    if (query[x + i] != target[y + i]) {
                        kputw(u, &str, d_buffer_ptr);
                        kputc(int2base[target[y+i]], &str, d_buffer_ptr);
                        ++n_mm; u = 0;
                    } else ++u;
                }
                x += len; y += len;
            } else if (op == 2) { // deletion
                if (k > 0 && k < n_cigar - 1) {
                    kputw(u, &str, d_buffer_ptr); kputc('^', &str, d_buffer_ptr);
                    for (int i = 0; i < len; ++i)
                        kputc(int2base[target[y+i]], &str, d_buffer_ptr);
                    u = 0; n_gap += len;
                }
                y += len;
            } else if (op == 1) x += len, n_gap += len; // insertion
        }
        kputw(u, &str, d_buffer_ptr); kputc(0, &str, d_buffer_ptr);
        NM = n_mm + n_gap;
        cigar = (uint32_t*)str.s;
    }
    // store to d_alns (needed downstream)
    mem_aln_t *a = &d_alns[seqID].a[alnID];
    a->cigar = cigar;
    a->n_cigar = n_cigar;
    a->score = score;
    a->NM = NM;

    // === FINALIZE PART ===
    mem_alnreg_t *ar = &d_regs[seqID].a[alnID];

    // Filter secondary alignments
    if (ar->secondary >= 0) {
        int skip = 0;
        if (ar->is_alt || !(d_opt->flag & MEM_F_ALL)) skip = 1;
        else if (ar->score < d_regs[seqID].a[ar->secondary].score * d_opt->drop_ratio) skip = 1;
        if (skip) {
            a->rid = -1;
            int basic_offset = d_offsets[seqID] + alnID;
            d_rids[basic_offset] = -1;
            return;
        }
    }

    // mapq
    if (ar->secondary<0) a->mapq = mem_approx_mapq_se(d_opt, ar);
    else a->mapq = 0;
    // calculate pos
    int is_rev; int64_t pos;
    pos = bns_depos_gpu(d_bns, d_seed_records[ID].reflen_left==0? ar->rb : ar->re-1, &is_rev);

    // fix cigar: squeeze out leading or trailing deletions
    if (n_cigar > 0) {
        if ((cigar[0]&0xf) == 2) {
            pos += cigar[0]>>4;
            --n_cigar;
            cigar = &cigar[1];
        } else if ((cigar[n_cigar-1]&0xf) == 2) {
            --n_cigar;
        }
    }
    // add clipping to cigar
    int seq_offset = d_seq_offset[seqID];
    int seq_offset_next = d_seq_offset[seqID + 1];
    int l_query = seq_offset_next - seq_offset;

    int qb = ar->qb; int qe = ar->qe;
    if (qb != 0 || qe != l_query) {
        int clip5, clip3;
        clip5 = is_rev? l_query - qe : qb;
        clip3 = is_rev? qb : l_query - qe;
        uint32_t *new_cigar = (uint32_t*)CUDAKernelMalloc(d_buffer_ptr, 4 * (n_cigar + 2), 4);
        if (clip5) {
            new_cigar[0] = clip5<<4 | 3;
            memcpy(&new_cigar[1], cigar, n_cigar*4);
            ++n_cigar;
        } else
            memcpy(new_cigar, cigar, n_cigar*4);
        if (clip3) {
            new_cigar[n_cigar++] = clip3<<4 | 3;
        }
        a->n_cigar = n_cigar;
        a->cigar = new_cigar;
        cigar = new_cigar;
    }

    // calculate rid, is_alt
    a->rid = bns_pos2rid_gpu(d_bns, pos);
    a->pos = pos - d_bns->anns[a->rid].offset;
    a->is_rev = is_rev;
    a->is_alt = ar->is_alt;
    a->alt_sc = ar->alt_sc;
    a->sub = ar->sub>ar->csub? ar->sub : ar->csub;

    // flag and sub
    int flag = 0;
    if (ar->secondary>=0) {
        a->sub = -1;
        flag |= 0x100;
    } else if (alnID>0){
        flag |= (d_opt->flag&MEM_F_NO_MULTI)? 0x10000 : 0x800;
    }
    if (ar->rid<0) flag |= 0x4;
    if (is_rev) flag |= 0x10;
    a->flag = flag;

    // pack output arrays
    int basic_offset = d_offsets[seqID] + alnID;
    d_rids[basic_offset] = a->rid;
    d_positions[basic_offset] = a->pos;
    d_ncigars[basic_offset] = a->n_cigar;
    d_flags[basic_offset] = a->flag;
    d_mapqs[basic_offset] = a->mapq;

    int cigar_offset = basic_offset * MAX_N_CIGAR;
    for(int kk=0; kk<a->n_cigar; kk++)
        d_cigars[cigar_offset + kk] = a->cigar[kk];
}

