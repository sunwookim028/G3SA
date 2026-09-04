#include "bwa.h"
#include "fmindex.cuh"

__device__ int devicehashK(const uint8_t* s){
    int out = 0;
    for (int i=0; i<KMER_K; i++){
        if (s[i]==4) return -1;
        out += s[i]*pow4(KMER_K-1-i);
    }
    return out;
}

#define \
GET_OCC_GPU(pp, c, occ_id_pp, y_pp, occ_pp, one_hot_bwt_str_c_pp, match_mask_pp) \
                int64_t occ_id_pp = pp >> CP_SHIFT; \
                int64_t y_pp = pp & CP_MASK; \
                int64_t occ_pp = d_cp_occ[occ_id_pp].cp_count[c]; \
                uint64_t one_hot_bwt_str_c_pp = d_cp_occ[occ_id_pp].one_hot_bwt_str[c]; \
                uint64_t match_mask_pp = one_hot_bwt_str_c_pp & d_one_hot[y_pp]; \
                occ_pp += __popcll(match_mask_pp);

#define \
GET_OCC2_GPU(pp, c, occ_id_pp, y_pp, occ_pp, one_hot_bwt_str_c_pp, match_mask_pp) \
                int64_t occ_id_pp = pp >> CP_SHIFT; \
                int64_t y_pp = pp & CP_MASK; \
                int64_t occ_pp = d_cp_occ2[occ_id_pp].cp_count[c]; \
                uint64_t one_hot_bwt_str_c_pp = d_cp_occ2[occ_id_pp].one_hot_bwt_str[c]; \
                uint64_t match_mask_pp = one_hot_bwt_str_c_pp & d_one_hot[y_pp]; \
                occ_pp += __popcll(match_mask_pp);


// b is guaranteed to be a base among A, C, G, T.
// LF mapping: LF(k) = C[b] + Occ(b, k-1), where Occ(b, k-1) counts b in BWT[0..k-1]
// (0-indexed inclusive up to k-1).  GET_OCC_GPU counts inclusively, so pass k-1.
// This matches backwardExt convention: count[b] + GET_OCC_GPU(sp, b) with sp = x[0]-1.
__device__ void LFMap(const fmindex_t *devFmIndex, uint64_t k, uint8_t b, uint64_t *bk)
{
    CP_OCC *d_cp_occ = devFmIndex->cpOcc;
    uint64_t *d_one_hot = devFmIndex->oneHot;
    int64_t *count = devFmIndex->count;

    GET_OCC_GPU(k-1, b, occ_id_k, y_k, occ_k, one_hot_bwt_str_b_k, match_mask_k);
    *bk = count[b] + occ_k;
}

__device__ void sa_lookup(const fmindex_t *devFmIndex, uint64_t k, int64_t *rb)
{
    uint32_t *suffixArrayLsWord = devFmIndex->suffixArrayLsWord;
    int8_t *suffixArrayMsByte = devFmIndex->suffixArrayMsByte;
    uint8_t *packedBwt = devFmIndex->packedBwt;
    int64_t sentinelIndex = *(devFmIndex->sentinelIndex);

    int offset = 0;
    uint8_t bwt_b;
    uint32_t packed_idx;
    uint8_t packed_offset;
    while(k & SA_COMPX_MASK)
    {
        if(k == sentinelIndex)
        {
            *rb = offset;
            return;
        } 

        // get bwt_b
        packed_idx = k >> 2;
        packed_offset = k & 0x3;
        bwt_b = packedBwt[packed_idx];
        for(uint8_t ii = 0; ii < 3 - packed_offset; ii++)
        {
            bwt_b = bwt_b >> 2;
        }
        bwt_b = bwt_b & 0x3;

        LFMap(devFmIndex, k, bwt_b, &k);
        offset++;
    }

    int64_t sa_entry = suffixArrayMsByte[k >> SA_COMPX];
    sa_entry = sa_entry << 32;
    sa_entry = sa_entry + suffixArrayLsWord[k >> SA_COMPX];
    sa_entry += offset;

    *rb = sa_entry;
}


__device__ void backwardExt(const int64_t sentinel_index, const bwtintv_t *smem, uint8_t base, bwtintv_t *nextSmem, const uint64_t *d_one_hot, const CP_OCC *d_cp_occ, const int64_t *d_count)
{
	uint8_t b;
	int64_t k[4], l[4], s[4];
	for(b = 0; b < 4; b++)
	{
		int64_t sp = (int64_t)(smem->x[0]) - 1;
		int64_t ep = (int64_t)(smem->x[0]) + (int64_t)(smem->x[2]) - 1;
		GET_OCC_GPU(sp, b, occ_id_sp, y_sp, occ_sp, one_hot_bwt_str_c_sp, match_mask_sp);
		GET_OCC_GPU(ep, b, occ_id_ep, y_ep, occ_ep, one_hot_bwt_str_c_ep, match_mask_ep);
		k[b] = d_count[b] + occ_sp;
		s[b] = occ_ep - occ_sp;
	}

	int64_t sentinel_offset = 0;
	if((smem->x[0] <= sentinel_index) && ((smem->x[0] + smem->x[2]) > sentinel_index)) sentinel_offset = 1;
	l[3] = smem->x[1] + sentinel_offset;
	l[2] = l[3] + s[3];
	l[1] = l[2] + s[2];
	l[0] = l[1] + s[1];

    nextSmem->x[0] = k[base];
    nextSmem->x[1] = l[base];
    nextSmem->x[2] = s[base];
    return;
}

__device__ void backwardExtBackward(const int64_t sentinel_index, const bwtintv_t *smem, uint8_t base, bwtintv_t *nextSmem, const uint64_t *d_one_hot, const CP_OCC *d_cp_occ, const int64_t *d_count)
{
    int64_t sp = (int64_t)(smem->x[0]) - 1;
    int64_t ep = (int64_t)(smem->x[0]) + (int64_t)(smem->x[2]) - 1;
    GET_OCC_GPU(sp, base, occ_id_sp, y_sp, occ_sp, one_hot_bwt_str_c_sp, match_mask_sp);
    GET_OCC_GPU(ep, base, occ_id_ep, y_ep, occ_ep, one_hot_bwt_str_c_ep, match_mask_ep);
    uint64_t newK = d_count[base] + occ_sp;
    uint64_t newS = occ_ep - occ_sp;

    nextSmem->x[0] = newK;
    nextSmem->x[2] = newS;
    return;
}

// SM OCC cache helpers are defined in fmindex.cuh (inline, guarded by G3_SM_OCC_CACHE).



/**
 *  extend 2 bases at once backward. also supports forward extend
 *  base pair order:
 *  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
 *  AA  AC  AG  AT  CA  CC  CG  CT  GA  GC  GG  GT  TA  TC  TG  TT
 *  idx: (base1, base0) 
 */
__device__ void backwardExt2(const int64_t sentinel_index, const bwtintv_t *smem, uint8_t base0, uint8_t base1, bwtintv_t *nextSmem, const uint64_t *d_one_hot, const CP_OCC *d_cp_occ, const int64_t *d_count, const CP_OCC2 *d_cp_occ2, const int64_t *d_count2, const uint8_t *d_first_base)
{
    uint8_t b;
    uint8_t basePair = base1 * 4 + base0;
    int64_t k[16], l[16], s[16];
    for (b = 0; b < 16; b++)
	{
		int64_t sp = (int64_t)(smem->x[0]) - 1;
		int64_t ep = (int64_t)(smem->x[0]) + (int64_t)(smem->x[2]) - 1;

		GET_OCC2_GPU(sp, b, occ_id_sp, y_sp, occ_sp, one_hot_bwt_str_c_sp, match_mask_sp);
		GET_OCC2_GPU(ep, b, occ_id_ep, y_ep, occ_ep, one_hot_bwt_str_c_ep, match_mask_ep);

		k[b] = d_count2[b] + occ_sp;
		s[b] = occ_ep - occ_sp;
	}

	int64_t sentinel_offset = 0;
	if((smem->x[0] <= sentinel_index) && ((smem->x[0] + smem->x[2]) > sentinel_index))
    {
       sentinel_offset = 1;
    }
	int64_t sentinel_offset2 = 0;
    uint8_t first_base = *d_first_base;
    bwtintv_t check2;
    backwardExt(sentinel_index, smem, first_base, &check2, d_one_hot, d_cp_occ, d_count);
    if((check2.x[0] <= sentinel_index) && (check2.x[0] + check2.x[2]) > sentinel_index)
    {
        sentinel_offset2 = 1;
    }

#define AA_ 0
#define AC_ 1
#define AG_ 2
#define AT_ 3
#define CA_ 4
#define CC_ 5
#define CG_ 6
#define CT_ 7
#define GA_ 8
#define GC_ 9
#define GG_ 10
#define GT_ 11
#define TA_ 12
#define TC_ 13
#define TG_ 14
#define TT_ 15
    l[TT_] = smem->x[1] + sentinel_offset;
    l[GT_] = l[TT_] + s[TT_];
    l[CT_] = l[GT_] + s[GT_];
    l[AT_] = l[CT_] + s[CT_];
    l[TG_] = l[AT_] + s[AT_];
    l[GG_] = l[TG_] + s[TG_];
    l[CG_] = l[GG_] + s[GG_];
    l[AG_] = l[CG_] + s[CG_];
    l[TC_] = l[AG_] + s[AG_];
    l[GC_] = l[TC_] + s[TC_];
    l[CC_] = l[GC_] + s[GC_];
    l[AC_] = l[CC_] + s[CC_];
    l[TA_] = l[AC_] + s[AC_];
    l[GA_] = l[TA_] + s[TA_];
    l[CA_] = l[GA_] + s[GA_];
    l[AA_] = l[CA_] + s[CA_];

    for (int jjj = first_base; jjj >= 0; jjj--)
    {
        for (int iii = 0; iii < 4; iii++)
        {
            b = iii * 4 + jjj;
            l[b] += sentinel_offset2;
        }
    }

    nextSmem->x[0] = k[basePair];
    nextSmem->x[1] = l[basePair];
    nextSmem->x[2] = s[basePair];

    return;
}

__device__ void backwardExt2Backward(const int64_t sentinel_index, const bwtintv_t *smem, uint8_t base0, uint8_t base1, bwtintv_t *nextSmem, const uint64_t *d_one_hot, const CP_OCC *d_cp_occ, const int64_t *d_count, const CP_OCC2 *d_cp_occ2, const int64_t *d_count2, const uint8_t *d_first_base)
{
    uint8_t basePair = base1 * 4 + base0;

    int64_t sp = (int64_t)(smem->x[0]) - 1;
    int64_t ep = (int64_t)(smem->x[0]) + (int64_t)(smem->x[2]) - 1;

    GET_OCC2_GPU(sp, basePair, occ_id_sp, y_sp, occ_sp, one_hot_bwt_str_c_sp, match_mask_sp);
    GET_OCC2_GPU(ep, basePair, occ_id_ep, y_ep, occ_ep, one_hot_bwt_str_c_ep, match_mask_ep);

    uint64_t nextK = d_count2[basePair] + occ_sp;
    uint64_t nextS = occ_ep - occ_sp;
    nextSmem->x[0] = nextK;
    nextSmem->x[2] = nextS;
    return;
}
