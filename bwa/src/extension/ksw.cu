/*
 * ksw.cu -- Smith-Waterman local/global alignment kernels.
 *
 * ksw_extend_warp2, g_defr, revseq, push_cigar, and ksw_align2 are adapted
 * from minhhpham/bwa (https://github.com/minhhpham/bwa), Copyright (c)
 * minhhpham, GPL-3.0 licensed — a GPU port of BWA-MEM (Copyright (c)
 * Dana-Farber Cancer Institute, Broad Institute, Genome Research Ltd.).
 * ksw_global3 (grid-tiled global alignment) is an independently-written
 * port of BWA-MEM2's ksw_global2. This file is part of G3SA and is
 * licensed under GPL-3.0 (see LICENSE).
 */

#include "ksw.cuh"
#include "macro.h"
#include "gmem_alloc.cuh"

__device__ const kswr_t g_defr = { 0, -1, -1, -1, -1, -1, -1 };

/*
 * ksw_i16_scalar: scalar Smith-Waterman local alignment (replaces m128i SSE-emulation).
 *
 * Produces identical scores/coordinates to the old ksw_i16 striped implementation.
 * Uses plain int16_t arrays; no m128i struct anywhere in the hot path.
 *
 * Memory layout: allocates H0[qlen], H1[qlen], E[qlen], Hmax[qlen] as int16_t arrays
 * from d_buffer_ptr, plus a query profile qp[m*qlen] and a score2 band array b[].
 * All values are clamped to [0, INT16_MAX] to match saturating unsigned arithmetic.
 */
static __device__ kswr_t ksw_i16_scalar(int qlen, const uint8_t *query, int tlen, const uint8_t *target,
	int m, const int8_t *mat,
	int _o_del, int _e_del, int _o_ins, int _e_ins, int xtra, void* d_buffer_ptr)
{
	int i, j, te = -1, gmax = 0, minsc, endsc;
	int m_b, n_b;
	uint64_t *b;
	kswr_t r;

	/* xtra flags */
	r = g_defr;
	minsc = (xtra & KSW_XSUBO) ? xtra & 0xffff : 0x10000;
	endsc = (xtra & KSW_XSTOP) ? xtra & 0xffff : 0x10000;
	/* Pre-allocate b with tlen capacity to eliminate reallocs in the score2 band loop.
	 * b[i] records rows where imax >= minsc; worst case one entry per target row. */
	m_b = tlen; n_b = 0;
	b = (tlen > 0)
		? (uint64_t*)CUDAKernelMalloc(d_buffer_ptr, (size_t)tlen * sizeof(uint64_t), 8)
		: nullptr;

	/* pre-computed gap costs */
	int16_t oe_del = (int16_t)(_o_del + _e_del);
	int16_t e_del  = (int16_t)_e_del;
	int16_t oe_ins = (int16_t)(_o_ins + _e_ins);
	int16_t e_ins  = (int16_t)_e_ins;

	/* find the max substitution score for score2 window calculation */
	int mat_max = 0;
	for (i = 0; i < m * m; ++i)
		if ((int)mat[i] > mat_max) mat_max = (int)mat[i];

	/* allocate working arrays */
	int16_t *H0   = (int16_t*)CUDAKernelMalloc(d_buffer_ptr, qlen * sizeof(int16_t), 2);
	int16_t *H1   = (int16_t*)CUDAKernelMalloc(d_buffer_ptr, qlen * sizeof(int16_t), 2);
	int16_t *E    = (int16_t*)CUDAKernelMalloc(d_buffer_ptr, qlen * sizeof(int16_t), 2);
	int16_t *Hmax = (int16_t*)CUDAKernelMalloc(d_buffer_ptr, qlen * sizeof(int16_t), 2);

	/* build query profile: qp[k*qlen + j] = mat[k*m + query[j]] */
	int8_t *qp = (int8_t*)CUDAKernelMalloc(d_buffer_ptr, m * qlen * sizeof(int8_t), 1);
	for (int k = 0; k < m; ++k) {
		const int8_t *row = &mat[k * m];
		for (j = 0; j < qlen; ++j)
			qp[k * qlen + j] = row[query[j]];
	}

	/* initialise H0, E, Hmax to 0 */
	for (j = 0; j < qlen; ++j) H0[j] = E[j] = Hmax[j] = 0;

	/* main DP loop: rows = target positions, cols = query positions */
	for (i = 0; i < tlen; ++i) {
		int16_t f = 0;
		int16_t imax = 0;
		const int8_t *S = &qp[target[i] * qlen]; /* score row for this target char */
		int16_t h_diag = 0; /* H0[j-1]: diagonal predecessor, wraps from 0 before j=0 */

		for (j = 0; j < qlen; ++j) {
			/* M = H0[j-1] + score (diagonal) */
			int16_t h_prev = h_diag; /* save before overwriting */
			h_diag = H0[j];          /* for next iteration's diagonal */

			int16_t h = (int16_t)((int16_t)h_prev + (int16_t)S[j]);
			if (h < 0) h = 0;

			/* h = max(M, E[j], F) */
			int16_t e = E[j];
			if (e > h) h = e;
			if (f > h) h = f;

			H1[j] = h;
			if (h > imax) imax = h;

			/* update E for next row: E[j] = max(E[j]-e_del, h-oe_del) clamped ≥ 0 */
			int16_t ne = (int16_t)(e - e_del); if (ne < 0) ne = 0;
			int16_t t  = (int16_t)(h - oe_del); if (t < 0) t = 0;
			E[j] = (ne > t) ? ne : t;

			/* update F for next col: f = max(f-e_ins, h-oe_ins) clamped ≥ 0 */
			int16_t nf = (int16_t)(f - e_ins); if (nf < 0) nf = 0;
			t = (int16_t)(h - oe_ins); if (t < 0) t = 0;
			f = (nf > t) ? nf : t;
		}

		/* track score2 band */
		if (imax >= minsc) {
			if (n_b == 0 || (int32_t)b[n_b-1] + 1 != i) {
				if (n_b == m_b) {
					/* Pre-allocated tlen entries should be sufficient.
					 * If not (edge case), fall back to the doubling realloc. */
					m_b = m_b ? m_b << 1 : 8;
					b = (uint64_t*)CUDAKernelRealloc(d_buffer_ptr, b, 8 * m_b, 8);
				}
				b[n_b++] = (uint64_t)(int32_t)imax << 32 | (uint32_t)i;
			} else if ((int)(b[n_b-1] >> 32) < (int)imax) {
				b[n_b-1] = (uint64_t)(int32_t)imax << 32 | (uint32_t)i;
			}
		}

		if (imax > gmax) {
			gmax = imax; te = i;
			for (j = 0; j < qlen; ++j) Hmax[j] = H1[j];
			if (gmax >= endsc) break;
		}

		/* swap rows */
		int16_t *tmp = H1; H1 = H0; H0 = tmp;
	}

	r.score = gmax; r.te = te;

	/* find qe: query position achieving gmax in Hmax row */
	{
		int hmax = -1;
		for (j = 0, r.qe = -1; j < qlen; ++j) {
			int v = (int)(uint16_t)Hmax[j];
			if (v > hmax) { hmax = v; r.qe = j; }
			else if (v == hmax && j < r.qe) r.qe = j;
		}
	}

	/* find score2/te2 */
	if (b && mat_max > 0) {
		int window = (gmax + mat_max - 1) / mat_max;
		int low = te - window, high = te + window;
		for (i = 0; i < n_b; ++i) {
			int e = (int32_t)b[i];
			if ((e < low || e > high) && (int)(b[i] >> 32) > r.score2)
				r.score2 = (int)(b[i] >> 32), r.te2 = e;
		}
	}

	return r;
}

__device__ static inline void revseq(int l, uint8_t *s)
{
	int i, t;
	for (i = 0; i < l>>1; ++i)
		t = s[i], s[i] = s[l - 1 - i], s[l - 1 - i] = t;
}

__device__ kswr_t ksw_align2(int qlen, uint8_t *query, int tlen, uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int xtra, void* d_buffer_ptr)
{
	kswr_t r, rr;
	r = ksw_i16_scalar(qlen, query, tlen, target, m, mat, o_del, e_del, o_ins, e_ins, xtra, d_buffer_ptr);
	if ((xtra&KSW_XSTART) == 0 || ((xtra&KSW_XSUBO) && r.score < (xtra&0xffff))) return r;
	revseq(r.qe + 1, query); revseq(r.te + 1, target); // +1 because qe/te points to the exact end, not the position after the end
	rr = ksw_i16_scalar(r.qe + 1, query, r.te + 1, target, m, mat, o_del, e_del, o_ins, e_ins, KSW_XSTOP | r.score, d_buffer_ptr);
	revseq(r.qe + 1, query); revseq(r.te + 1, target);
	if (r.score == rr.score)
		r.tb = r.te - rr.te, r.qb = r.qe - rr.qe;
	return r;
}


typedef struct {
	int32_t h, e;
} eh_t;

/* scoring of 2 characters given scoring matrix mat, and dimension m*/
__device__ static inline int score(uint8_t A, uint8_t B, const int8_t *mat, int m){
	return (int)mat[A*m+B];
}
#define ALL_THREADS 0xffffffff  // mask indicating all threads participate in shuffle instruction
#define max2(a,b) ((a)>(b)?(a):(b))

/********************
 * Global alignment *
 ********************/
 #define MINUS_INF -0x40000000

__device__ static inline uint32_t *push_cigar(int *n_cigar, int *m_cigar, uint32_t *cigar, int op, int len, void* d_buffer_ptr)
{
	if (*n_cigar == 0 || op != (cigar[(*n_cigar) - 1]&0xf)) {
		if (*n_cigar == *m_cigar) {
			*m_cigar = *m_cigar? (*m_cigar)<<1 : 4;
			cigar = (uint32_t*)CUDAKernelRealloc(d_buffer_ptr, cigar, (*m_cigar) << 2, 4);
		}
		cigar[(*n_cigar)++] = len<<4 | op;
	} else cigar[(*n_cigar)-1] += len<<4;
	return cigar;
}

__device__ int ksw_global2(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int w, int *n_cigar_, uint32_t **cigar_, void* d_buffer_ptr)
{
	eh_t *eh;
	int8_t *qp; // query profile
	int i, j, k, oe_del = o_del + e_del, oe_ins = o_ins + e_ins, score, n_col;
	uint8_t *z; // backtrack matrix; in each cell: f<<4|e<<2|h; in principle, we can halve the memory, but backtrack will be a little more complex
	if (n_cigar_) *n_cigar_ = 0;
	// allocate memory
	n_col = qlen < 2*w+1? qlen : 2*w+1; // maximum #columns of the backtrack matrix
	z = n_cigar_ && cigar_? (uint8_t*)CUDAKernelMalloc(d_buffer_ptr, (long)n_col * tlen, 1) : 0;
	qp = (int8_t*)CUDAKernelMalloc(d_buffer_ptr, qlen * m, 1);
	eh = (eh_t*)CUDAKernelCalloc(d_buffer_ptr, qlen + 1, 8, 4);
	// generate the query profile
	for (k = i = 0; k < m; ++k) {
		const int8_t *p = &mat[k * m];
		for (j = 0; j < qlen; ++j) qp[i++] = p[query[j]];
	}
	// fill the first row
	eh[0].h = 0; eh[0].e = MINUS_INF;
	for (j = 1; j <= qlen && j <= w; ++j)
		eh[j].h = -(o_ins + e_ins * j), eh[j].e = MINUS_INF;
	for (; j <= qlen; ++j) eh[j].h = eh[j].e = MINUS_INF; // everything is -inf outside the band
	// DP loop
	for (i = 0; i < tlen; ++i) { // target sequence is in the outer loop
		int32_t f = MINUS_INF, h1, beg, end, t;
		int8_t *q = &qp[target[i] * qlen];
		beg = i > w? i - w : 0;
		end = i + w + 1 < qlen? i + w + 1 : qlen; // only loop through [beg,end) of the query sequence
		h1 = beg == 0? -(o_del + e_del * (i + 1)) : MINUS_INF;
		if (n_cigar_ && cigar_) {
			uint8_t *zi = &z[(long)i * n_col];
			for (j = beg; j < end; ++j) {
				// At the beginning of the loop: eh[j] = { H(i-1,j-1), E(i,j) }, f = F(i,j) and h1 = H(i,j-1)
				// Cells are computed in the following order:
				//   M(i,j)   = H(i-1,j-1) + S(i,j)
				//   H(i,j)   = max{M(i,j), E(i,j), F(i,j)}
				//   E(i+1,j) = max{M(i,j)-gapo, E(i,j)} - gape
				//   F(i,j+1) = max{M(i,j)-gapo, F(i,j)} - gape
				// We have to separate M(i,j); otherwise the direction may not be recorded correctly.
				// However, a CIGAR like "10M3I3D10M" allowed by local() is disallowed by global().
				// Such a CIGAR may occur, in theory, if mismatch_penalty > 2*gap_ext_penalty + 2*gap_open_penalty/k.
				// In practice, this should happen very rarely given a reasonable scoring system.
				eh_t *p = &eh[j];
				int32_t h, m = p->h, e = p->e;
				uint8_t d; // direction
				p->h = h1;
				m += q[j];
				d = m >= e? 0 : 1;
				h = m >= e? m : e;
				d = h >= f? d : 2;
				h = h >= f? h : f;
				h1 = h;
				t = m - oe_del;
				e -= e_del;
				d |= e > t? 1<<2 : 0;
				e  = e > t? e    : t;
				p->e = e;
				t = m - oe_ins;
				f -= e_ins;
				d |= f > t? 2<<4 : 0; // if we want to halve the memory, use one bit only, instead of two
				f  = f > t? f    : t;
				zi[j - beg] = d; // z[i,j] keeps h for the current cell and e/f for the next cell
			}
		} else {
			for (j = beg; j < end; ++j) {
				eh_t *p = &eh[j];
				int32_t h, m = p->h, e = p->e;
				p->h = h1;
				m += q[j];
				h = m >= e? m : e;
				h = h >= f? h : f;
				h1 = h;
				t = m - oe_del;
				e -= e_del;
				e  = e > t? e : t;
				p->e = e;
				t = m - oe_ins;
				f -= e_ins;
				f  = f > t? f : t;
			}
		}
		eh[end].h = h1; eh[end].e = MINUS_INF;
	}
	score = eh[qlen].h;
	if (n_cigar_ && cigar_) { // backtrack
		/* Pre-allocate MAX_N_CIGAR entries — covers all practical 150bp reads
		 * and avoids push_cigar reallocs (which leak the old allocation). */
		int n_cigar = 0, m_cigar = MAX_N_CIGAR, which = 0;
		uint32_t *cigar = (uint32_t*)CUDAKernelMalloc(d_buffer_ptr, m_cigar*sizeof(uint32_t), 4);
		uint32_t tmp;
		i = tlen - 1; k = (i + w + 1 < qlen? i + w + 1 : qlen) - 1; // (i,k) points to the last cell
		while (i >= 0 && k >= 0) {
			which = z[(long)i * n_col + (k - (i > w? i - w : 0))] >> (which<<1) & 3;
			if (which == 0)      cigar = push_cigar(&n_cigar, &m_cigar, cigar, 0, 1, d_buffer_ptr), --i, --k;
			else if (which == 1) cigar = push_cigar(&n_cigar, &m_cigar, cigar, 2, 1, d_buffer_ptr), --i;
			else                 cigar = push_cigar(&n_cigar, &m_cigar, cigar, 1, 1, d_buffer_ptr), --k;
		}
		if (i >= 0) cigar = push_cigar(&n_cigar, &m_cigar, cigar, 2, i + 1, d_buffer_ptr);
		if (k >= 0) cigar = push_cigar(&n_cigar, &m_cigar, cigar, 1, k + 1, d_buffer_ptr);
		for (i = 0; i < n_cigar>>1; ++i) // reverse CIGAR
			tmp = cigar[i], cigar[i] = cigar[n_cigar-1-i], cigar[n_cigar-1-i] = tmp;
		*n_cigar_ = n_cigar, *cigar_ = cigar;
	}
	return score;
}
/* Banded SW extension; threads outside the band are idle (not warp-utilization-optimal). */
__device__ int ksw_extend_warp2(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int w, int end_bonus, int h0, int *_qle, int *_tle, int *_gtle, int *_gscore, int *_max_off)
{
	if (qlen>KSW_MAX_QLEN){return 0;}
	__shared__	int16_t SM_H[KSW_MAX_QLEN], SM_E[KSW_MAX_QLEN];
	__shared__ int8_t SM_mat[25]; // scoring matrix in shared memory (5x5, read-only)
	if (threadIdx.x < 25) SM_mat[threadIdx.x] = mat[threadIdx.x];
	__syncwarp();
	int oe_del = o_del + e_del;
	int oe_ins = o_ins + e_ins;
	int e, f, h;
	int e1_;
	int h1_, h_1, h11;
	int max_score = h0;	// best score
	int i_m=-1, j_m=-1;	// position of best score
	int max_gscore = 0; // score of end-to-end alignment
	int i_gscore = -1;	// position of best end-to-end alignment score
	int max_=0, max_ins, max_del, k;

/* PART I */
	// first row scoring
	SM_H[0] = h0;
	for (int j=threadIdx.x+1; j<qlen+1; j+=WARPSIZE){	// j is col index
		SM_E[j] = 0;
		h = h0 - oe_ins - (j-1)*e_ins;
		SM_H[j] = (h>0)? h : 0;
	}

/* PART 2 */
	k = m*m;
	for (int i = 0; i < k; i++) {
		max_ = max_ > SM_mat[i] ? max_ : SM_mat[i];
	}
	max_ins = (int)((double)(qlen * max_ + end_bonus - o_ins) / e_ins + 1.);
	max_ins = max_ins > 1? max_ins : 1;
	w = w < max_ins? w : max_ins;
	max_del = (int)((double)(qlen * max_ + end_bonus - o_del) / e_del + 1.);
	max_del = max_del > 1? max_del : 1;
	w = w < max_del? w : max_del;

/* PART 3 */
	// first we fill the top-left corner where we don't have enough parallelism
	for (int anti_diag=0; anti_diag<WARPSIZE-1; anti_diag++){
		int i = threadIdx.x; 				// row index on the matrix
		int j = anti_diag - threadIdx.x;	// col index on the matrix
		int M, t;

		e1_ = __shfl_up_sync(ALL_THREADS, e, 1);
		h1_ = __shfl_up_sync(ALL_THREADS, h, 1);
		h11 = __shfl_up_sync(ALL_THREADS, h_1, 1);
		h_1 = h;

		if (j == 0) {
			h_1 = h0 - (o_del + e_del*(i+1));
			if (h_1 < 0) h_1 = 0;
			f = 0;	// first column of F
		}

		if (threadIdx.x == 0) {
			e1_ = SM_E[j+1];
			h1_ = SM_H[j+1];
			h11 = SM_H[j];
		}

		if ((0 <= i) && (i < tlen) && (0 <= j) && (j < qlen) && (i-w<=j) && (j<=i+w+1)) {
			M = h11;
			M = M ? M + score(target[i], query[j], SM_mat, m) : 0;
			h = max2(M, e1_);
			h = max2(h, f);
			if (h > max_score) {
				max_score = h;
				i_m = i;
				j_m = j;
			}
			else if (h == max_score) {
				if ((i == i_m) && (j > j_m))  {
					i_m = i;
					j_m = j;
				}
				else if ((i_m < 0) && (j_m < 0)) {
					i_m = i;
					j_m = j;
				}
			}
			if (j == qlen - 1) {  // reached the last query column: track end-to-end score
				if (h > max_gscore) { max_gscore = h; i_gscore = i; }
			}
			t = M - oe_del;
			t = max2(t, 0);
			e = e1_ - e_del;
			e = max2(e, t);

			t = M - oe_ins;
			t = max2(t, 0);
			f -= e_ins;
			f = max2(f, t);

			if (threadIdx.x==WARPSIZE-1) {
				SM_H[j] = h;
				SM_E[j] = e;
			}
		}
		else {
			h = 0;
			e = 0;
			f = 0;
		}

	}

	// fill the rest of the matrix where we have enough parallelism
	int Ntile = ceil((float)tlen/WARPSIZE);
	int qlen_padded = qlen>=32? qlen : 32;	// pad qlen so that we have correct overflow for small matrix
	for (int tile_ID=0; tile_ID<Ntile; tile_ID++){	// tile loop
		int i, j;
		i = tile_ID*WARPSIZE + threadIdx.x;	// row index on matrix
		j = (WARPSIZE-1) - threadIdx.x; 		// col index
		for (int anti_diag=0; anti_diag<(tile_ID+1)*WARPSIZE+w+2 && anti_diag<qlen_padded; anti_diag++){	// anti-diagonal loop
			if ((j>=qlen_padded) || j>(tile_ID+1)*WARPSIZE+w+1){			// when hit the end of this tile, overflow to next tile
				i = i+WARPSIZE;		// over flow to its row on the next tile
				j = 0;
			}

			int M, t;

			e1_ = __shfl_up_sync(ALL_THREADS, e, 1);
			h1_ = __shfl_up_sync(ALL_THREADS, h, 1);
			h11 = __shfl_up_sync(ALL_THREADS, h_1, 1);
			h_1 = h;

			if (j == 0) {
				h_1 = h0 - (o_del + e_del*(i+1));
				if (h_1 < 0) h_1 = 0;
				f = 0;	// first column of F
			}

			if (threadIdx.x == 0) {
				e1_ = SM_E[j];
				h1_ = SM_H[j+1];
				h11 = SM_H[j];
			}

			if ((0 <= i) && (i < tlen) && (0 <= j) && (j < qlen) && (i-w<=j)) {
				M = h11;
				M = M ? M + score(target[i], query[j], SM_mat, m) : 0;
				h = max2(M, e1_);
				h = max2(h, f);

				if (h > max_score) {
					max_score = h;
					i_m = i;
					j_m = j;
				}
				else if (h == max_score) {
					if ((i == i_m))  {
						i_m = i;
						j_m = j;
					}
					else if ((i_m < 0) && (j_m < 0)) {
						i_m = i;
						j_m = j;
					}
				}
				if (j == qlen - 1) {  // reached the last query column: track end-to-end score
					if (h > max_gscore) { max_gscore = h; i_gscore = i; }
				}
				t = M - oe_del;
				t = max2(t, 0);
				e = e1_ - e_del;
				e = max2(e, t);

				t = M - oe_ins;
				t = max2(t, 0);
				f -= e_ins;
				f = max2(f, t);

				if (threadIdx.x==WARPSIZE-1) {
					SM_H[j] = h_1;
					SM_E[j] = e;
				}
			}
			j++;

		}
	}
	// finished filling the matrix, now we find the max of max_score across the warp
	// use reduction to find the max of 32 max's
	for (int i=0; i<5; i++){
		int tmp = __shfl_down_sync(ALL_THREADS, max_score, 1<<i);
		int tmp_i = __shfl_down_sync(ALL_THREADS, i_m, 1<<i);
		int tmp_j = __shfl_down_sync(ALL_THREADS, j_m, 1<<i);
		if (max_score < tmp) {max_score = tmp; i_m = tmp_i; j_m = tmp_j;}
		else if (max_score == tmp) {
			if (tmp_i < i_m) {
				i_m = tmp_i;
				j_m = tmp_j;
			}
			else if ((tmp_i == i_m) && (tmp_j > j_m))  {
				i_m = tmp_i;
				j_m = tmp_j;
			}
		}
		tmp = __shfl_down_sync(ALL_THREADS, max_gscore, 1<<i);
		tmp_i = __shfl_down_sync(ALL_THREADS, i_gscore, 1<<i);
		if (max_gscore < tmp){max_gscore = tmp; i_gscore = tmp_i;}
	}

	// write max, i_m, j_m to global memory
	if (_qle) *_qle = j_m + 1;
	if (_tle) *_tle = i_m + 1;
	if (_gtle) *_gtle = i_gscore + 1;
	if (_gscore) *_gscore = max_gscore;
	// max_off: diagonal offset at the best-score position (approximation: |j_m - i_m|).
	// Thread 0 has the correct i_m/j_m after reduction; other threads write stale values,
	// but local_extend broadcasts from thread 0 via __shfl_sync.
	if (_max_off) *_max_off = (i_m >= 0 && j_m >= 0) ? abs(j_m - i_m) : 0;
	return max_score;	// only thread 0's result is valid
}

/********************
 * Global alignment *
 ********************/
 /* MINUS_INF already defined above */

 /* SW global executing at warp level
	BLOCKSIZE = WARPSIZE = 32
	requires at least qlen*4 bytes of shared memory
	currently implemented at 500*4 bytes of shared mem	
	return max score in the matrix, coordinates of max score, traceback matrix (we don't do traceback here because it's inefficient at warp level, should do at thread level)
	NOTATIONS:
		SM_H[], SM_E: shared memory arrays for storing H and E of thread 31 for transitioning between tiles
		e, f, h     : E[i,j], F[i,j], H[i,j] to be calculated in an iteration
		e1_			: E[i-1,j] during a cell calculation
		h1_,h_1,h11 : // H[i-1,j], H[i,j-1], H[i-1,j-1]
		max_score   : the max score that a thread has found
		i_m, j_m	: the position where we found max_score
		traceback   : traceback matrix. traceback[i,j] 	= 0 if score came from [i-1, j-1]	(cigar match)
														= 1 if score came from [i  , j-1]	(cigar insert to target)
														= 2 if score came from [i-1, j  ]	(cigar delete from target)
					traceback has to be allocated prior to calling this function, with at least tlen*qlen, row-major order
	CALCULATION:
		E[i,j] = max(H[i-1,j]-gap_open_penalty, E[i-1,j]-gap_ext_penalty)
		F[i,j] = max(H[i,j-1]-gap_open_penalty, F[i,j-1]-gap_ext_penalty)
		H[i,j] = max(0, E[i,j], F[i,j], H[i-1.j-1]+score(query[j],target[i]))
 */

#define GRIDSIZE 16
__device__ int ksw_global3(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int w, int *n_cigar_, uint32_t **cigar_, void* d_buffer_ptr)
{
	int32_t SM_h[2*GRIDSIZE-1];	// thread-local arrays for border values during traceback
	int32_t SM_e[GRIDSIZE];	// thread-local arrays for border values during traceback
	int32_t SM_f[GRIDSIZE];	// thread-local arrays for border values during traceback
	uint8_t direction[(GRIDSIZE)*(GRIDSIZE)];

	eh_t *eh;
	int8_t *qp; // query profile
	int i, j, k, oe_del = o_del + e_del, oe_ins = o_ins + e_ins, score;
	if (n_cigar_) *n_cigar_ = 0;
	
	int gridIdx_i, gridIdx_j, gridIdx;
	unsigned long n_gridcol = max((int)ceil((float)(qlen)/(GRIDSIZE)), 0);
	unsigned long n_gridrow = max((int)ceil((float)(tlen)/(GRIDSIZE)), 0);
	unsigned long n_grid = n_gridcol*n_gridrow;

	int32_t *border_h = (int32_t*)CUDAKernelMalloc(d_buffer_ptr, (2*GRIDSIZE+1)*n_grid*sizeof(int32_t), 1);
	int32_t *border_e = (int32_t*)CUDAKernelMalloc(d_buffer_ptr, (GRIDSIZE+1)*n_grid*sizeof(int32_t), 1);
	int32_t *border_f = (int32_t*)CUDAKernelMalloc(d_buffer_ptr, (GRIDSIZE+1)*n_grid*sizeof(int32_t), 1);

	qp = (int8_t*)CUDAKernelMalloc(d_buffer_ptr, qlen * m, 1);
	eh = (eh_t*)CUDAKernelCalloc(d_buffer_ptr, qlen + 1, 8, 4);
	
	// generate the query profile
	for (k = i = 0; k < m; ++k) {
		const int8_t *p = &mat[k * m];
		for (j = 0; j < qlen; ++j) qp[i++] = p[query[j]];
	}
	// fill the first row
	eh[0].h = 0; eh[0].e = MINUS_INF;
	for (j = 1; j <= qlen && j <= w; ++j)
		eh[j].h = -(o_ins + e_ins * j), eh[j].e = MINUS_INF;
	for (; j <= qlen; ++j) eh[j].h = eh[j].e = MINUS_INF; // everything is -inf outside the band
	// DP loop
    if(target != 0) {
	for (i = 0; i < tlen; ++i) { // target sequence is in the outer loop
		int32_t f = MINUS_INF, h1, t;
		int beg, end;
		int8_t *q = &qp[target[i] * qlen];
		beg = i > w? i - w : 0;
		end = i + w + 1 < qlen? i + w + 1 : qlen; // only loop through [beg,end) of the query sequence
		
		h1 = beg == 0? -(o_del + e_del * (i + 1)) : MINUS_INF;
		if (n_cigar_ && cigar_) {
			for (j = beg; j < end; ++j) {
				// At the beginning of the loop: eh[j] = { H(i-1,j-1), E(i,j) }, f = F(i,j) and h1 = H(i,j-1)
				// Cells are computed in the following order:
				//   M(i,j)   = H(i-1,j-1) + S(i,j)
				//   H(i,j)   = max{M(i,j), E(i,j), F(i,j)}
				//   E(i+1,j) = max{M(i,j)-gapo, E(i,j)} - gape
				//   F(i,j+1) = max{M(i,j)-gapo, F(i,j)} - gape
				// We have to separate M(i,j); otherwise the direction may not be recorded correctly.
				// However, a CIGAR like "10M3I3D10M" allowed by local() is disallowed by global().
				// Such a CIGAR may occur, in theory, if mismatch_penalty > 2*gap_ext_penalty + 2*gap_open_penalty/k.
				// In practice, this should happen very rarely given a reasonable scoring system.

				eh_t *p = &eh[j];
				int32_t h, m = p->h, e = p->e;
				uint8_t d; // direction
				// save score if the cell is on the grid border
				if ((i % GRIDSIZE == 0) || (j % GRIDSIZE == 0)) {
					// idx of the grid
					gridIdx_i = i/GRIDSIZE;
					gridIdx_j = j/GRIDSIZE;
					gridIdx = (gridIdx_i*n_gridcol + gridIdx_j);
					int borderIdx;	// idx of the cell within the grid
					if ((i%GRIDSIZE != 0) && (j%GRIDSIZE == 0)) {		// column
						border_f[gridIdx*(GRIDSIZE) + (i%GRIDSIZE)] = f;
						borderIdx = (i%GRIDSIZE)+GRIDSIZE-1;
					}
					else if ((i%GRIDSIZE == 0) && (j%GRIDSIZE != 0)) {		// row
						border_e[gridIdx*(GRIDSIZE) + (j%GRIDSIZE)] = e;
						borderIdx = (j%GRIDSIZE);
					}
					else {	// corner
						border_f[gridIdx*(GRIDSIZE) + (i%GRIDSIZE)] = f;
						border_e[gridIdx*(GRIDSIZE) + (j%GRIDSIZE)] = e;
						
						borderIdx = (j%GRIDSIZE);
					}

					border_h[gridIdx*(2*GRIDSIZE-1) + borderIdx] = m;
				}
				p->h = h1;
				m += q[j];
				d = m >= e? 0 : 1;
				h = m >= e? m : e;
				d = h >= f? d : 2;
				h = h >= f? h : f;
				h1 = h;

				t = m - oe_del;
				e -= e_del;
				d |= e > t? 1<<2 : 0;
				e  = e > t? e    : t;
				p->e = e;
				t = m - oe_ins;
				f -= e_ins;
				d |= f > t? 2<<4 : 0; // if we want to halve the memory, use one bit only, instead of two
				f  = f > t? f    : t;
			}
			
		} else {
			for (j = beg; j < end; ++j) {
				eh_t *p = &eh[j];
				int32_t h, m = p->h, e = p->e;
				p->h = h1;
				m += q[j];
				h = m >= e? m : e;
				h = h >= f? h : f;
				h1 = h;
				t = m - oe_del;
				e -= e_del;
				e  = e > t? e : t;
				p->e = e;
				t = m - oe_ins;
				f -= e_ins;
				f  = f > t? f : t;
			}
		}
		eh[end].h = h1; eh[end].e = MINUS_INF;
		
	}
    }
	score = eh[qlen].h;

	if (n_cigar_ && cigar_) { // backtrack
		/* Pre-allocate MAX_N_CIGAR entries — covers all practical 150bp reads
		 * and avoids push_cigar reallocs (which leak the old allocation). */
		int n_cigar = 0, m_cigar = MAX_N_CIGAR, which = 0;
		uint32_t *cigar = (uint32_t*)CUDAKernelMalloc(d_buffer_ptr, m_cigar*sizeof(uint32_t), 4);
		uint32_t tmp;

		i = tlen - 1; k = (i + w + 1 < qlen? i + w + 1 : qlen) - 1; // (i,k) points to the last cell

		gridIdx_i = (i/GRIDSIZE);
		gridIdx_j = (k/GRIDSIZE);

		int cornerIdx_i;
		int cornerIdx_j;
		int gridIdx;

		while (i >= 0 && k >= 0) {
			cornerIdx_i = gridIdx_i*GRIDSIZE;
			cornerIdx_j = gridIdx_j*GRIDSIZE;
			gridIdx = gridIdx_i*n_gridcol + gridIdx_j;
			// load global border data to thread-local arrays.
			// Check band boundaries to avoid reading uninitialized values for
			// out-of-band cells (eliminates run-to-run non-determinism from
			// stale data in the uninitialized buffer pool).
			{
				int ci = cornerIdx_i;
				int cj = cornerIdx_j;
				// SM_h[0..GRIDSIZE-1] = row border (i=ci, j=cj+idx)
				for (int idx = 0; idx < GRIDSIZE; idx++) {
					int bi = ci, bj = cj + idx;
					if (bi >= 0 && bj >= 0 && bi < tlen && bj < qlen
						&& bj >= bi - w && bj <= bi + w)
						SM_h[idx] = border_h[(gridIdx)*(2*GRIDSIZE-1) + idx];
					else
						SM_h[idx] = 0;
				}
				// SM_h[GRIDSIZE..2*GRIDSIZE-2] = col border (i=ci+ii, j=cj)
				for (int idx = GRIDSIZE; idx < 2*GRIDSIZE-1; idx++) {
					int ii = idx - GRIDSIZE + 1;
					int bi = ci + ii, bj = cj;
					if (bi >= 0 && bj >= 0 && bi < tlen && bj < qlen
						&& bj >= bi - w && bj <= bi + w)
						SM_h[idx] = border_h[(gridIdx)*(2*GRIDSIZE-1) + idx];
					else
						SM_h[idx] = 0;
				}
				for (int idx = 0; idx < GRIDSIZE; idx++) {
					int bi = ci, bj = cj + idx;
					if (bi >= 0 && bj >= 0 && bi < tlen && bj < qlen
						&& bj >= bi - w && bj <= bi + w)
						SM_e[idx] = border_e[gridIdx*(GRIDSIZE)+idx];
					else
						SM_e[idx] = 0;
				}
				for (int idx = 0; idx < GRIDSIZE; idx++) {
					int bi = ci + idx, bj = cj;
					if (bi >= 0 && bj >= 0 && bi < tlen && bj < qlen
						&& bj >= bi - w && bj <= bi + w)
						SM_f[idx] = border_f[gridIdx*(GRIDSIZE)+idx];
					else
						SM_f[idx] = 0;
				}
			}

			// grid filling
			int i_, j_;
			for (i_ = 0; i_+cornerIdx_i <= i; i_++) {
				int8_t *q = &qp[target[i_+cornerIdx_i] * qlen];

				int32_t h;
				int32_t h11 = SM_h[0];						// H[i-1, j-1]
				int32_t h10 = SM_h[1];						// H[i-1, j]
				int32_t h01 = SM_h[i_ + GRIDSIZE];		// H[i,   j-1]
				int32_t e = SM_e[0];
				int32_t f = SM_f[i_];

				for (j_ = 0; j_+cornerIdx_j <= k; j_++) {
					if ((i_+cornerIdx_i+w <= j_+cornerIdx_j) || (i_+cornerIdx_i > j_+cornerIdx_j + w)) {
						f = MINUS_INF;
						h11 = SM_h[j_+1];
						continue;
					}
					if (i_+cornerIdx_i == j_+cornerIdx_j + w) {f = MINUS_INF;}
					h10 = SM_h[j_+1];
					e = SM_e[j_];


					int32_t m = h11;
					uint8_t d; // direction
					m += q[j_+cornerIdx_j];
					d = m >= e? 0 : 1;
					h = m >= e? m : e;
					d = h >= f? d : 2;
					h = h >= f? h : f;
					
					// update h11, h01 for next iteration
					SM_h[j_] = h01;
					h11 = h10;
					h01 = h;

					int t = m - oe_del;
					e -= e_del;
					d |= e > t? 1<<2 : 0;
					e  = e > t? e    : t;
					SM_e[j_] = e;
					t = m - oe_ins;
					f -= e_ins;
					d |= f > t? 2<<4 : 0; // if we want to halve the memory, use one bit only, instead of two
					f  = f > t? f    : t;
					direction[(i_)*GRIDSIZE + (j_)] = d;
					
				}
			}

			// grid traceback
			i_ = i - cornerIdx_i;
			j_ = k - cornerIdx_j;
			
			while((i_ >= 0) && (j_ >= 0) && (i_+cornerIdx_i >= 0) && (j_+cornerIdx_j >= 0)) {
				which = direction[(i_)*GRIDSIZE + (j_)] >> (which<<1) & 3;
				if (which == 0)      cigar = push_cigar(&n_cigar, &m_cigar, cigar, 0, 1, d_buffer_ptr), --i_, --j_;
				else if (which == 1) cigar = push_cigar(&n_cigar, &m_cigar, cigar, 2, 1, d_buffer_ptr), --i_;
				else                 cigar = push_cigar(&n_cigar, &m_cigar, cigar, 1, 1, d_buffer_ptr), --j_;
			}

			
			i = i_ + cornerIdx_i;
			k = j_ + cornerIdx_j;

			if (i_ < 0) {gridIdx_i--;}
			if (j_ < 0) {gridIdx_j--;}
		}

		if (i >= 0) cigar = push_cigar(&n_cigar, &m_cigar, cigar, 2, i, d_buffer_ptr);
		if (k >= 0) cigar = push_cigar(&n_cigar, &m_cigar, cigar, 1, k, d_buffer_ptr);
		for (i = 0; i < n_cigar>>1; ++i) // reverse CIGAR
			tmp = cigar[i], cigar[i] = cigar[n_cigar-1-i], cigar[n_cigar-1-i] = tmp;
		*n_cigar_ = n_cigar, *cigar_ = cigar;
	}
	return score;
}

