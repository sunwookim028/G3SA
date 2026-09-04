/*
 * Portions of this file are adapted from minhhpham/bwa
 * (https://github.com/minhhpham/bwa), Copyright (c) minhhpham,
 * GPL-3.0 licensed — a GPU port of BWA (Copyright (c) Genome Research Ltd.,
 * Broad Institute, Dana-Farber Cancer Institute). This file is part of
 * G3SA and is licensed under GPL-3.0 (see LICENSE).
 */

#ifndef LOAD_KMER_INDEX_H
#define LOAD_KMER_INDEX_H

#include "hash_kmer.h"

kmers_bucket_t *loadKMerIndex(const char* path);

#endif
