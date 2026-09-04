/*
 * Portions of this file are adapted from minhhpham/bwa
 * (https://github.com/minhhpham/bwa), Copyright (c) minhhpham,
 * GPL-3.0 licensed — a GPU port of BWA (Copyright (c) Genome Research Ltd.,
 * Broad Institute, Dana-Farber Cancer Institute). This file is part of
 * G3SA and is licensed under GPL-3.0 (see LICENSE).
 */

#ifndef HASH_KMER_INDEX_H
#define HASH_KMER_INDEX_H

#define KMER_K 12

#include "bwt.h"

#define pow4(x) (1<<(2*(x)))  // 4^x

typedef struct kmers_bucket_t {
	bwtint_t x[3]; // same as first 3 elements on bwtintv_t
	// bwtintv_t.info not included here because it contains length of match, which is always KMER_K in this case
} kmers_bucket_t;


#ifdef __cplusplus
extern "C"{
#endif

	kmers_bucket_t *loadKMerIndex(const char* path);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif
