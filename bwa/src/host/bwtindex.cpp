/* The MIT License

   Copyright (c) 2008 Genome Research Ltd (GRL).
   Modified Copyright (C) 2019 Intel Corporation, Heng Li.

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <zlib.h>
#include <string>
#include <vector>
#include <fstream>
#include "bntseq.h"
#include "bwa.h"
#include "bwt.h"
#include "utils.h"
#include "fmi_search.h"
#include "sais.h"
#include "hash_kmer.h"
#include "datadump.h"

#define OCC_INTERVAL 0x80

// Read .pac file and reconstruct the nucleotide reference as a std::string
static void pac2nt_local(const char *prefix, std::string &ref_seq)
{
	char pac_file[PATH_MAX];
	snprintf(pac_file, PATH_MAX, "%s.pac", prefix);

	// Compute sequence length: same formula as FMI_search::pac_seq_len
	FILE *fp = fopen(pac_file, "rb");
	if (!fp) { fprintf(stderr, "[E::%s] cannot open %s\n", __func__, pac_file); exit(1); }
	fseek(fp, -1, SEEK_END);
	long file_pos = ftell(fp);
	unsigned char last_byte;
	fread(&last_byte, 1, 1, fp);
	int64_t pac_len = (file_pos - 1) * 4 + (int)last_byte;
	// Read packed data
	int64_t pac_size = (pac_len >> 2) + ((pac_len & 3) ? 1 : 0);
	fseek(fp, 0, SEEK_SET);
	std::vector<unsigned char> pac_buf(pac_size);
	fread(pac_buf.data(), 1, pac_size, fp);
	fclose(fp);
	// Unpack 2-bit encoded bases
	ref_seq.resize(pac_len);
	const char decode[4] = {'A', 'C', 'G', 'T'};
	for (int64_t i = 0; i < pac_len; i++) {
		ref_seq[i] = decode[(pac_buf[i >> 2] >> ((3 - (i & 3)) << 1)) & 3];
	}
}

// Build legacy BWA .bwt and .sa files from the .pac reference.
// This makes the index compatible with the mem command's bwa_idx_load_bwt().
static void build_legacy_bwt_sa(const char *prefix, int64_t l_pac)
{
	clock_t t = clock();
	fprintf(stderr, "[bwa_index] Building legacy .bwt/.sa... ");

	// 1. Read forward reference from .pac, then append reverse complement
	std::string fwd_seq;
	pac2nt_local(prefix, fwd_seq);
	int64_t fwd_len = fwd_seq.length();

	// BWA BWT is built on forward + reverse complement (2*l_pac)
	std::string ref_seq;
	ref_seq.resize(fwd_len * 2);
	for (int64_t i = 0; i < fwd_len; i++)
		ref_seq[i] = fwd_seq[i];
	for (int64_t i = 0; i < fwd_len; i++) {
		char c = fwd_seq[fwd_len - 1 - i];
		switch (c) {
		case 'A': ref_seq[fwd_len + i] = 'T'; break;
		case 'C': ref_seq[fwd_len + i] = 'G'; break;
		case 'G': ref_seq[fwd_len + i] = 'C'; break;
		case 'T': ref_seq[fwd_len + i] = 'A'; break;
		default:  ref_seq[fwd_len + i] = 'A'; break;
		}
	}
	int64_t seq_len = ref_seq.length(); // = 2 * fwd_len

	// 2. Build suffix array on the concatenated sequence
	int64_t *sa = (int64_t *)malloc((seq_len + 2) * sizeof(int64_t));
	sa[0] = seq_len;
	saisxx(ref_seq.c_str(), sa + 1, (int64_t)seq_len);

	// 3. Construct raw BWT (2-bit packed, 16 bases per uint32_t)
	// BWA convention: skip the $ position (primary), store seq_len characters
	int64_t raw_bwt_size = (seq_len + 15) / 16; // in uint32_t
	uint32_t *raw_bwt = (uint32_t *)calloc(raw_bwt_size, sizeof(uint32_t));
	bwtint_t primary = 0;
	bwtint_t L2[5] = {0, 0, 0, 0, 0};
	bwtint_t counts[4] = {0, 0, 0, 0};
	int64_t bwt_idx = 0;
	for (int64_t i = 0; i <= seq_len; i++) {
		if (sa[i] == 0) {
			primary = i;
			continue; // skip $ position
		}
		int c;
		switch (ref_seq[sa[i] - 1]) {
		case 'A': c = 0; break;
		case 'C': c = 1; break;
		case 'G': c = 2; break;
		case 'T': c = 3; break;
		default:  c = 0; break;
		}
		counts[c]++;
		// Pack: 16 bases per uint32_t, MSB first
		raw_bwt[bwt_idx >> 4] |= (uint32_t)c << ((~bwt_idx & 0xf) << 1);
		bwt_idx++;
	}
	// bwt_idx should equal seq_len now

	// Compute L2 (cumulative counts)
	L2[0] = 0;
	L2[1] = counts[0];
	L2[2] = counts[0] + counts[1];
	L2[3] = counts[0] + counts[1] + counts[2];
	L2[4] = seq_len;

	// 4. Interleave occurrence counts into the BWT (bwt_bwtupdate_core equivalent)
	// Format: for each 128-char block: 8 uint32_t occ + ceil(chars/16) uint32_t BWT
	// Full blocks have 8+8=16 uint32_t. The last block may be shorter.
	// Plus a final 8 uint32_t occ checkpoint after the last block.
	int64_t n_full_blocks = seq_len / OCC_INTERVAL;
	int64_t remainder = seq_len % OCC_INTERVAL;
	int64_t last_block_words = remainder ? (remainder + 15) / 16 : 0;
	int64_t new_bwt_size = n_full_blocks * 16 + (remainder ? (8 + last_block_words) : 0) + 8;
	uint32_t *new_bwt = (uint32_t *)calloc(new_bwt_size, sizeof(uint32_t));

	bwtint_t occ[4] = {0, 0, 0, 0};
	int64_t raw_pos = 0; // position in raw_bwt (uint32_t index)
	int64_t new_pos = 0; // position in new_bwt (uint32_t index)
	int64_t n_chars_done = 0;

	while (n_chars_done < seq_len) {
		// Write occurrence counts at this checkpoint
		memcpy(new_bwt + new_pos, occ, sizeof(bwtint_t) * 4);
		new_pos += 8; // 4 bwtint_t = 8 uint32_t

		// How many chars and BWT words in this block
		int64_t chars_in_block = seq_len - n_chars_done;
		if (chars_in_block > OCC_INTERVAL) chars_in_block = OCC_INTERVAL;
		int64_t words_in_block = (chars_in_block + 15) / 16;

		// Copy exactly the needed BWT words
		for (int64_t w = 0; w < words_in_block; w++) {
			new_bwt[new_pos++] = raw_bwt[raw_pos++];
		}

		// Update occurrence counts for this block
		for (int64_t j = 0; j < chars_in_block; j++) {
			int64_t abs_pos = n_chars_done + j;
			int c = (raw_bwt[abs_pos >> 4] >> ((~abs_pos & 0xf) << 1)) & 3;
			occ[c]++;
		}
		n_chars_done += chars_in_block;
	}
	// Write final occurrence counts checkpoint
	memcpy(new_bwt + new_pos, occ, sizeof(bwtint_t) * 4);

	// 5. Build bwt_t and dump
	bwt_t bwt;
	memset(&bwt, 0, sizeof(bwt_t));
	bwt.primary = primary;
	memcpy(bwt.L2, L2, sizeof(L2));
	bwt.seq_len = seq_len;
	bwt.bwt_size = new_bwt_size;
	bwt.bwt = new_bwt;
	bwt_gen_cnt_table(&bwt);

	char fname[PATH_MAX];
	snprintf(fname, PATH_MAX, "%s.bwt", prefix);
	bwt_dump_bwt(fname, &bwt);

	// 6. Compute sampled suffix array and dump
	bwt_cal_sa(&bwt, 32); // SA sample interval = 32 (BWA default)

	snprintf(fname, PATH_MAX, "%s.sa", prefix);
	bwt_dump_sa(fname, &bwt);

	free(new_bwt);
	free(bwt.sa);
	free(sa);

	fprintf(stderr, "%.2f sec\n", (float)(clock() - t) / CLOCKS_PER_SEC);
}

static int bwa_idx_build(const char *fa, const char *prefix)
{
	clock_t t;
	int64_t l_pac;

	{ // nucleotide indexing
		gzFile fp = xzopen(fa, "r");
		t = clock();
		fprintf(stderr, "[bwa_index] Pack FASTA... ");
		l_pac = bns_fasta2bntseq(fp, prefix, 1);
		fprintf(stderr, "%.2f sec\n", (float)(clock() - t) / CLOCKS_PER_SEC);
		err_gzclose(fp);
		FMI_search *fmi = new FMI_search(prefix);
		fmi->build_index();
		delete fmi;
	}

	// Generate legacy BWA .bwt and .sa files for the mem command
	build_legacy_bwt_sa(prefix, l_pac);

	// Build and save KMER hash table for fast reseed init
	{
		clock_t t_hash = clock();
		fprintf(stderr, "[bwa_index] Building KMER-K=%d hash table... ", KMER_K);
		// Load the BWT that build_legacy_bwt_sa() just wrote to disk.
		char bwt_path[PATH_MAX];
		snprintf(bwt_path, PATH_MAX, "%s.bwt", prefix);
		bwt_t *bwt_loaded = bwt_restore_bwt(bwt_path);
		kmers_bucket_t *hashTable = createHashKTable(bwt_loaded);
		bwt_destroy(bwt_loaded);
		char hash_path[PATH_MAX];
		snprintf(hash_path, PATH_MAX, "%s.hash", prefix);
		dumpArray(hashTable, (unsigned long long)pow4(KMER_K), std::string(hash_path));
		free(hashTable);
		fprintf(stderr, "%.2f sec\n", (float)(clock() - t_hash) / CLOCKS_PER_SEC);
	}

	return 0;
}

int bwa_index(int argc, char *argv[])
{
	int c;
	char *prefix = 0;
	while ((c = getopt(argc, argv, "p:")) >= 0) {
		if (c == 'p') prefix = optarg;
		else return 1;
	}

	if (optind + 1 > argc) {
		fprintf(stderr, "Usage: g3 index [-p prefix] <in.fasta>\n");
		return 1;
	}
	if (prefix == 0) prefix = argv[optind];
	bwa_idx_build(argv[optind], prefix);
	return 0;
}
