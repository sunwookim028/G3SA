#ifndef KSW_CUH
#define KSW_CUH

#include "stdint.h"
#include "bwa.h"

#define KSW_XSTOP  0x20000
#define KSW_XSUBO  0x40000
#define KSW_XSTART 0x80000
#define KSW_MAX_QLEN 500

typedef struct {
	int score; // best score
	int te, qe; // target end and query end
	int score2, te2; // second best score and ending position on the target
	int tb, qb; // target start and query start
} kswr_t;

__device__ kswr_t ksw_align2(int qlen, uint8_t *query, int tlen, uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int xtra, void* d_buffer_ptr);

__device__ int ksw_global2(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int w, int *n_cigar_, uint32_t **cigar_, void* d_buffer_ptr);

/* Warp-cooperative SW local extension (one warp per alignment). */
__device__ int ksw_extend_warp2(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int w, int end_bonus, int h0, int *_qle, int *_tle, int *_gtle, int *_gscore, int *_max_off);
__device__ int ksw_global3(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int w, int *n_cigar_, uint32_t **cigar_, void* d_buffer_ptr);
#endif
