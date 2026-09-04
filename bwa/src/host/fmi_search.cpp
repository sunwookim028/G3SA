/*************************************************************************************
                           The MIT License

   BWA-MEM2  (Sequence alignment using Burrows-Wheeler Transform),
   Copyright (C) 2019  Intel Corporation, Heng Li.

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

Authors: Sanchit Misra <sanchit.misra@intel.com>; Vasimuddin Md <vasimuddin.md@intel.com>;
*****************************************************************************************/

#include <iostream>
#include <stdio.h>
#include <cstdio>
#include "sais.h"
#include <x86intrin.h>
#include "fmi_search.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "safe_str_lib.h"
#ifdef __cplusplus
}
#endif

FMI_search::FMI_search(const char *fname)
{
    strcpy_s(file_name, PATH_MAX, fname);
    reference_seq_len = 0;
    sentinel_index = 0;
    index_alloc = 0;
    sa_ls_word = NULL;
    sa_ms_byte = NULL;
    cp_occ = NULL;
    one_hot_mask_array = NULL;
}

FMI_search::~FMI_search()
{
    if(sa_ms_byte)
        _mm_free(sa_ms_byte);
    if(sa_ls_word)
        _mm_free(sa_ls_word);
    if(cp_occ)
        _mm_free(cp_occ);
    if(cp_occ2)
        _mm_free(cp_occ2);
    if(one_hot_mask_array)
        _mm_free(one_hot_mask_array);
}

int64_t FMI_search::pac_seq_len(const char *fn_pac)
{
	FILE *fp;
	int64_t pac_len;
	uint8_t c;
	fp = xopen(fn_pac, "rb");
	err_fseek(fp, -1, SEEK_END);
	pac_len = err_ftell(fp);
	err_fread_noeof(&c, 1, 1, fp);
	err_fclose(fp);
	return (pac_len - 1) * 4 + (int)c;
}

void FMI_search::pac2nt(const char *fn_pac, std::string &reference_seq)
{
	uint8_t *buf2;
	int64_t i, pac_size, seq_len;
	FILE *fp;

	// initialization
	seq_len = pac_seq_len(fn_pac);
    assert(seq_len > 0);
    assert(seq_len <= 0x7fffffffffL);
	fp = xopen(fn_pac, "rb");

	// prepare sequence
	pac_size = (seq_len>>2) + ((seq_len&3) == 0? 0 : 1);
	buf2 = (uint8_t*)calloc(pac_size, 1);
    assert(buf2 != NULL);
	err_fread_noeof(buf2, 1, pac_size, fp);
	err_fclose(fp);
	for (i = 0; i < seq_len; ++i) {
		int nt = buf2[i>>2] >> ((3 - (i&3)) << 1) & 3;
        switch(nt)
        {
            case 0:
                reference_seq += "A";
            break;
            case 1:
                reference_seq += "C";
            break;
            case 2:
                reference_seq += "G";
            break;
            case 3:
                reference_seq += "T";
            break;
            default:
                fprintf(stderr, "ERROR! Value of nt is not in 0,1,2,3!");
                exit(EXIT_FAILURE);
        }
	}
    for(i = seq_len - 1; i >= 0; i--)
    {
        char c = reference_seq[i];
        switch(c)
        {
            case 'A':
                reference_seq += "T";
            break;
            case 'C':
                reference_seq += "G";
            break;
            case 'G':
                reference_seq += "C";
            break;
            case 'T':
                reference_seq += "A";
            break;
        }
    }
	free(buf2);
}

int FMI_search::build_fm_index(const char *ref_file_name, char *binary_seq, int64_t ref_seq_len, int64_t *sa_bwt)
{
    char outname[PATH_MAX];
// "prefix.counts"
    strcpy_s(outname, PATH_MAX, ref_file_name);
    strcat_s(outname, PATH_MAX, ".counts");
    std::fstream outCountsStream (outname, std::ios::out | std::ios::binary);
    outCountsStream.seekg(0);

    ref_seq_len++; // to include the sentinel
    outCountsStream.write((char *)(&ref_seq_len), 1 * sizeof(int64_t));
    outCountsStream.write((char*)count, 5 * sizeof(int64_t));
    outCountsStream.write((char*)count2, 17 * sizeof(int64_t));
    outCountsStream.close();
// end of "prefix.counts"

    uint8_t *bwt;
    uint8_t *bwt2;

    int64_t i;
    int64_t ref_seq_len_aligned = ((ref_seq_len + CP_BLOCK_SIZE - 1) / CP_BLOCK_SIZE) * CP_BLOCK_SIZE;
    int64_t size = ref_seq_len_aligned * sizeof(uint8_t);
    bwt = (uint8_t *)_mm_malloc(size, 64);
    assert_not_null(bwt, size, index_alloc);
    bwt2 = (uint8_t *)_mm_malloc(size, 64);
    assert_not_null(bwt2, size, index_alloc);

    sentinel_index = -1;
    int64_t sentinel2_index = -1;

    cp_occ_size = (ref_seq_len >> CP_SHIFT) + 1;
    fprintf(stderr, "Building Occ1 and Occ2 tables each of %ld rows.\n", cp_occ_size);


    for(i=0; i< ref_seq_len; i++)
    {
        if(sa_bwt[i] == 0)
        {
            bwt[i] = 4; // <- This represents the virtual '$'
            fprintf(stderr, "BWT[%ld] = 4\n", i);
            sentinel_index = i;
        }
#if 1
        else
        {
            char c = binary_seq[sa_bwt[i]-1];
            switch(c)
            {
                case 0: bwt[i] = 0;
                          break;
                case 1: bwt[i] = 1;
                          break;
                case 2: bwt[i] = 2;
                          break;
                case 3: bwt[i] = 3;
                          break;
                default:
                        fprintf(stderr, "ERROR! i = %ld, c = %c\n", i, c);
                        exit(EXIT_FAILURE);
            }
        }

        if(sa_bwt[i] == 1)
        {
            bwt2[i] = 4;
            sentinel2_index = i;
            fprintf(stderr, "BWT2[%ld] = 4\n", i);
        }
        else
        {
            char c;
            if (sa_bwt[i] == 0)
            {
                c = binary_seq[ref_seq_len - 2];
            }
            else 
            {
                c = binary_seq[sa_bwt[i]-2];
            }
            switch(c)
            {
                case 0: bwt2[i] = 0;
                          break;
                case 1: bwt2[i] = 1;
                          break;
                case 2: bwt2[i] = 2;
                          break;
                case 3: bwt2[i] = 3;
                          break;
                default:
                        fprintf(stderr, "ERROR! (BWT2) i = %ld, c = %c\n", i, c);
                        exit(EXIT_FAILURE);
            }
        }
#endif
    }
    for(i = ref_seq_len; i < ref_seq_len_aligned; i++)
    {
        bwt[i] = DUMMY_CHAR;
        bwt2[i] = DUMMY_CHAR;
    }

    fprintf(stderr, "sentinel_index = %ld\n", sentinel_index);

    // create checkpointed occ
    cp_occ = NULL;
    cp_occ2 = NULL;

    size = cp_occ_size * sizeof(CP_OCC);
    cp_occ = (CP_OCC *)_mm_malloc(size, 64);
    assert_not_null(cp_occ, size, index_alloc);
    memset(cp_occ, 0, cp_occ_size * sizeof(CP_OCC));
    cp_occ2 = (CP_OCC2 *)_mm_malloc(cp_occ_size * sizeof(CP_OCC2), 64);
    assert_not_null(cp_occ2, size, index_alloc);
    memset(cp_occ2, 0, cp_occ_size * sizeof(CP_OCC2));

    int64_t cp_count[16];
    memset(cp_count, 0, 16 * sizeof(int64_t));
    int64_t cp_count2[16];
    memset(cp_count2, 0, 16 * sizeof(int64_t));
    for(i = 0; i < ref_seq_len; i++)
    {
        if((i & CP_MASK) == 0)
        {
            int32_t k;
            CP_OCC cpo;
            for (k=0; k<4; k++)
            {
                cpo.cp_count[k] = cp_count[k];
                cpo.one_hot_bwt_str[k] = 0;
            }
            CP_OCC2 cpo2;
            for (k=0; k<16; k++)
            {
                cpo2.cp_count[k] = cp_count2[k];
                cpo2.one_hot_bwt_str[k] = 0;
            }

			int32_t j;
			for(j = 0; j < CP_BLOCK_SIZE; j++)
			{
                for (k=0; k<4; k++)
                {
                    cpo.one_hot_bwt_str[k] = cpo.one_hot_bwt_str[k] << 1;
                }
                for (k=0; k<16; k++)
                {
                    cpo2.one_hot_bwt_str[k] = cpo2.one_hot_bwt_str[k] << 1;
                }
				uint8_t c = bwt[i + j];
				uint8_t c2 = bwt2[i + j];
                if(c < 4)
                {
                    cpo.one_hot_bwt_str[c] += 1;
                    if (c2 < 4)
                    {
                        cpo2.one_hot_bwt_str[c2 * 4 + c] += 1;
                    }
                }
			}
            cp_occ[i >> CP_SHIFT] = cpo;
            cp_occ2[i >> CP_SHIFT] = cpo2;
        }
        if (i != sentinel_index)
        {
            cp_count[bwt[i]]++;
            if (i != sentinel2_index)
            {
                cp_count2[bwt2[i] * 4 + bwt[i]]++;
            }
        }
    }

// "prefix.packed.bwt" — 2-bit packed BWT for sa_lookup() on GPU
    {
        int64_t packedSize = ref_seq_len_aligned / 4 * sizeof(uint8_t);
        uint8_t *packed_bwt_buf = (uint8_t *)_mm_malloc(packedSize, 64);
        memset(packed_bwt_buf, 0, packedSize);
        for (int64_t pi = 0; pi < ref_seq_len; pi++) {
            uint8_t c = (bwt[pi] < 4) ? bwt[pi] : 0x3; // sentinel -> 0x3
            int64_t idx_p = pi >> 2;
            packed_bwt_buf[idx_p] = (packed_bwt_buf[idx_p] << 2) | c;
        }
        strcpy_s(outname, PATH_MAX, ref_file_name);
        strcat_s(outname, PATH_MAX, ".packed.bwt");
        std::fstream outPackedBwtStream(outname, std::ios::out | std::ios::binary);
        outPackedBwtStream.seekg(0);
        outPackedBwtStream.write((char*)packed_bwt_buf, packedSize);
        outPackedBwtStream.close();
        _mm_free(packed_bwt_buf);
    }
// end of "prefix.packed.bwt"

// "prefix.occ1"
    strcpy_s(outname, PATH_MAX, ref_file_name);
    strcat_s(outname, PATH_MAX, ".occ1");
    std::fstream outOcc1Stream (outname, std::ios::out | std::ios::binary);
    outOcc1Stream.seekg(0);

    outOcc1Stream.write((char*)cp_occ, cp_occ_size * sizeof(CP_OCC));
    _mm_free(cp_occ); cp_occ = nullptr;
    _mm_free(bwt); bwt = nullptr;
    outOcc1Stream.close();
// end of "prefix.occ1"


// "prefix.occ2"
    strcpy_s(outname, PATH_MAX, ref_file_name);
    strcat_s(outname, PATH_MAX, ".occ2");
    std::fstream outOcc2Stream (outname, std::ios::out | std::ios::binary);
    outOcc2Stream.seekg(0);	

    outOcc2Stream.write((char*)cp_occ2, cp_occ_size * sizeof(CP_OCC2));
    _mm_free(cp_occ2); cp_occ2 = nullptr;
    _mm_free(bwt2); bwt2 = nullptr;
    outOcc2Stream.close();
// end of "prefix.occ2"

// "prefix.sa"
    strcpy_s(outname, PATH_MAX, ref_file_name);
    strcat_s(outname, PATH_MAX, ".sa.v2");
    std::fstream outSaStream (outname, std::ios::out | std::ios::binary);
    outSaStream.seekg(0);	

    outSaStream.write((const char*)(&first_base), 1 * sizeof(uint8_t));
    outSaStream.write((char *)(&sentinel_index), 1 * sizeof(int64_t));

#ifdef SA_COMPRESSION
    size = ((ref_seq_len >> SA_COMPX)+ 1)  * sizeof(uint32_t);
    uint32_t *sa_ls_word = (uint32_t *)_mm_malloc(size, 64);
    assert_not_null(sa_ls_word, size, index_alloc);
    size = ((ref_seq_len >> SA_COMPX) + 1) * sizeof(int8_t);
    int8_t *sa_ms_byte = (int8_t *)_mm_malloc(size, 64);
    assert_not_null(sa_ms_byte, size, index_alloc);
    int64_t pos = 0;
    for(i = 0; i < ref_seq_len; i++)
    {
        if ((i & SA_COMPX_MASK) == 0)
        {
            sa_ls_word[pos] = sa_bwt[i] & 0xffffffff;
            sa_ms_byte[pos] = (sa_bwt[i] >> 32) & 0xff;
            pos++;
        }
    }
    fprintf(stderr, "compressed SA length: %ld, compressed ref_seq_len__: %ld\n", pos, ref_seq_len >> SA_COMPX);
    outSaStream.write((char*)sa_ms_byte, ((ref_seq_len >> SA_COMPX) + 1) * sizeof(int8_t));
    outSaStream.write((char*)sa_ls_word, ((ref_seq_len >> SA_COMPX) + 1) * sizeof(uint32_t));
    
#else

    size = ref_seq_len * sizeof(uint32_t);
    uint32_t *sa_ls_word = (uint32_t *)_mm_malloc(size, 64);
    assert_not_null(sa_ls_word, size, index_alloc);
    size = ref_seq_len * sizeof(int8_t);
    int8_t *sa_ms_byte = (int8_t *)_mm_malloc(size, 64);
    assert_not_null(sa_ms_byte, size, index_alloc);
    for(i = 0; i < ref_seq_len; i++)
    {
        sa_ls_word[i] = sa_bwt[i] & 0xffffffff;
        sa_ms_byte[i] = (sa_bwt[i] >> 32) & 0xff;
    }
    outSaStream.write((char*)sa_ms_byte, ref_seq_len * sizeof(int8_t));
    outSaStream.write((char*)sa_ls_word, ref_seq_len * sizeof(uint32_t));
    
#endif

    fprintf(stderr, "max_occ_ind = %ld\n", i >> CP_SHIFT);    
    fflush(stdout);

    _mm_free(sa_ms_byte); sa_ms_byte = nullptr;
    _mm_free(sa_ls_word); sa_ls_word = nullptr;

    outSaStream.close();
// end of "prefix.sa"
    return 0;
}

int FMI_search::build_index() {

    char *prefix = file_name;
    uint64_t startTick;
    startTick = __rdtsc();
    index_alloc = 0;

    std::string reference_seq;
    char pac_file_name[PATH_MAX];
    strcpy_s(pac_file_name, PATH_MAX, prefix);
    strcat_s(pac_file_name, PATH_MAX, ".pac");

    // read from .pac to generate std::string reference_seq
    // we do not concatenate the complementary strand.
    pac2nt(pac_file_name, reference_seq); 
	int64_t pac_len = reference_seq.length();
    int status;
    int64_t size = pac_len * sizeof(char);

    // generate ACTG -> 0123 representation of the reference seq
    // and store it in .0123 file.
    // Counts tables are also computed.
    char *binary_ref_seq = (char *)_mm_malloc(size, 64);
    index_alloc += size;
    assert_not_null(binary_ref_seq, size, index_alloc);
    char binary_ref_name[PATH_MAX];
    strcpy_s(binary_ref_name, PATH_MAX, prefix);
    strcat_s(binary_ref_name, PATH_MAX, ".0123");
    std::fstream binary_ref_stream (binary_ref_name, std::ios::out | std::ios::binary);
    binary_ref_stream.seekg(0);
    fprintf(stderr, "init ticks = %llu\n", __rdtsc() - startTick);
    startTick = __rdtsc();
    int64_t i;
	memset(count, 0, sizeof(int64_t) * 5);
    for(i = 0; i < pac_len; i++)
    {
        switch(reference_seq[i])
        {
            case 'A':
            binary_ref_seq[i] = 0, ++count[0];
            break;
            case 'C':
            binary_ref_seq[i] = 1, ++count[1];
            break;
            case 'G':
            binary_ref_seq[i] = 2, ++count[2];
            break;
            case 'T':
            binary_ref_seq[i] = 3, ++count[3];
            break;
            default:
            binary_ref_seq[i] = 4;
        }
    }
    first_base = (uint8_t)binary_ref_seq[0];
    fprintf(stderr, "First base = %d\n", first_base);
    fprintf(stderr, "Raw count = %ld, %ld, %ld, %ld, %ld\n", count[0], count[1], count[2], count[3], count[4]);
    count[4]=count[0]+count[1]+count[2]+count[3];
    count[3]=count[0]+count[1]+count[2];
    count[2]=count[0]+count[1];
    count[1]=count[0];
    count[0]=0;
    fprintf(stderr, "count = %ld, %ld, %ld, %ld, %ld\n", count[0], count[1], count[2], count[3], count[4]);

	memset(count2, 0, sizeof(int64_t) * 17);
    for(i = 0; i < pac_len - 1; i++)
    {
        switch(reference_seq[i])
        {
            case 'A':
                switch(reference_seq[i + 1])
                {
                    case 'A': ++count2[0]; break;
                    case 'C': ++count2[1]; break;
                    case 'G': ++count2[2]; break;
                    case 'T': ++count2[3]; break;
                    default: fprintf(stderr, "ERROR: unexpected base encoding\n");
                }
                break;
            case 'C':
                switch(reference_seq[i + 1])
                {
                    case 'A': ++count2[4]; break;
                    case 'C': ++count2[5]; break;
                    case 'G': ++count2[6]; break;
                    case 'T': ++count2[7]; break;
                    default: fprintf(stderr, "ERROR: unexpected base encoding\n");
                }
                break;
            case 'G':
                switch(reference_seq[i + 1])
                {
                    case 'A': ++count2[8]; break;
                    case 'C': ++count2[9]; break;
                    case 'G': ++count2[10]; break;
                    case 'T': ++count2[11]; break;
                    default: fprintf(stderr, "ERROR: unexpected base encoding\n");
                }
                break;
            case 'T':
                switch(reference_seq[i + 1])
                {
                    case 'A': ++count2[12]; break;
                    case 'C': ++count2[13]; break;
                    case 'G': ++count2[14]; break;
                    case 'T': ++count2[15]; break;
                    default: fprintf(stderr, "ERROR: unexpected base encoding\n");
                }
                break;
            default:;
        }
    }

    fprintf(stderr, "Raw count2 = %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld\n", \
                    count2[0], count2[1], count2[2], count2[3], count2[4],\
                              count2[5], count2[6], count2[7], count2[8],\
                              count2[9], count2[10], count2[11], count2[12],\
                              count2[13], count2[14], count2[15], count2[16]);
    for (i = 16; i > 0; i--)
    {
        int64_t sum = 0;
        for (int ii = 0; ii < i; ii++)
        {
            sum += count2[ii];
        }
        count2[i] = sum;
    }
    count2[0] = 0;

    fprintf(stderr, "count2 = %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld\n", \
            count2[0], count2[1], count2[2], count2[3], count2[4],\
            count2[5], count2[6], count2[7], count2[8],\
            count2[9], count2[10], count2[11], count2[12],\
            count2[13], count2[14], count2[15], count2[16]);


    fprintf(stderr, "ref seq len = %ld\n", pac_len); // No longer 2x ref_len
    binary_ref_stream.write(binary_ref_seq, pac_len * sizeof(char));
    fprintf(stderr, "binary seq ticks = %llu\n", __rdtsc() - startTick);
    startTick = __rdtsc();

    size = (pac_len + 2) * sizeof(int64_t);

    int64_t *suffix_array=(int64_t *)_mm_malloc(size, 64);
    index_alloc += size;
    assert_not_null(suffix_array, size, index_alloc);
    startTick = __rdtsc();
	status = saisxx(reference_seq.c_str(), suffix_array + 1, pac_len);
	suffix_array[0] = pac_len;
    fprintf(stderr, "build suffix-array ticks = %llu\n", __rdtsc() - startTick);
    startTick = __rdtsc();

	build_fm_index(prefix, binary_ref_seq, pac_len, suffix_array);
    fprintf(stderr, "build fm-index ticks = %llu\n", __rdtsc() - startTick);
    return 0;
}

void FMI_search::load_index()
{
    // oneHot
    one_hot_mask_array = (uint64_t *)malloc(64 * sizeof(uint64_t));
    uint64_t base = 0x8000000000000000L;
    one_hot_mask_array[0] = base;
    int64_t i;
    for(i = 1; i < 64; i++)
    {
        one_hot_mask_array[i] = (one_hot_mask_array[i - 1] >> 1) | base;
    }


    char *ref_file_name = file_name;
    char index_file_name[PATH_MAX];
    FILE *occ1Stream = NULL;
    FILE *occ2Stream = NULL;
    FILE *saStream = NULL;
    FILE *countsStream = NULL;
    FILE *packedBwtStream = NULL;

    // Read the counts 
    strcpy_s(index_file_name, PATH_MAX, ref_file_name);
    strcat_s(index_file_name, PATH_MAX, ".counts");
    countsStream = fopen(index_file_name,"rb");
    if (countsStream == NULL)
    {
        fprintf(stderr, "ERROR! Unable to open the file: %s\n", index_file_name);
        exit(EXIT_FAILURE);
    }
    else
    {
        fprintf(stderr, "* Index file found. Loading index from %s\n", index_file_name);
    }

    err_fread_noeof(&reference_seq_len, sizeof(int64_t), 1, countsStream);
    assert(reference_seq_len > 0);
    assert(reference_seq_len <= 0x7fffffffffL);
    fprintf(stderr, "* Reference seq len for bi-index = %ld\n", reference_seq_len);

    err_fread_noeof(&count[0], sizeof(int64_t), 5, countsStream);
    err_fread_noeof(&count2[0], sizeof(int64_t), 17, countsStream);
    fclose(countsStream);

    cp_occ_size = (reference_seq_len >> CP_SHIFT) + 1;
    cp_occ = NULL;
    // Load occ1
    strcpy_s(index_file_name, PATH_MAX, ref_file_name);
    strcat_s(index_file_name, PATH_MAX, ".occ1");
    occ1Stream = fopen(index_file_name,"rb");
    if (occ1Stream == NULL)
    {
        fprintf(stderr, "ERROR! Unable to open the file: %s\n", index_file_name);
        exit(EXIT_FAILURE);
    }
    else
    {
        fprintf(stderr, "* Index file found. Loading index from %s\n", index_file_name);
    }

    if ((cp_occ = (CP_OCC *)_mm_malloc(cp_occ_size * sizeof(CP_OCC), 64)) == NULL) {
        fprintf(stderr, "ERROR! unable to allocated cp_occ memory\n");
        exit(EXIT_FAILURE);
    }
    err_fread_noeof(cp_occ, sizeof(CP_OCC), cp_occ_size, occ1Stream);
    fclose(occ1Stream);


    // Load occ2
    strcpy_s(index_file_name, PATH_MAX, ref_file_name);
    strcat_s(index_file_name, PATH_MAX, ".occ2");
    occ2Stream = fopen(index_file_name,"rb");
    if (occ2Stream == NULL)
    {
        fprintf(stderr, "ERROR! Unable to open the file: %s\n", index_file_name);
        exit(EXIT_FAILURE);
    }
    else
    {
        fprintf(stderr, "* Index file found. Loading index from %s\n", index_file_name);
    }

    if((cp_occ2 = (CP_OCC2 *)_mm_malloc(cp_occ_size * sizeof(CP_OCC2), 64)) == NULL) {
        fprintf(stderr, "ERROR! unable to allocated cp_occ2 memory\n");
        exit(EXIT_FAILURE);
    }
    err_fread_noeof(cp_occ2, sizeof(CP_OCC2), cp_occ_size, occ2Stream);
    fclose(occ2Stream);
    strcpy_s(index_file_name, PATH_MAX, ref_file_name);
    strcat_s(index_file_name, PATH_MAX, ".sa.v2");
    saStream = fopen(index_file_name,"rb");
    if (saStream == NULL)
    {
        fprintf(stderr, "ERROR! Unable to open the file: %s\n", index_file_name);
        exit(EXIT_FAILURE);
    }
    else
    {
        fprintf(stderr, "* Index file found. Loading index from %s\n", index_file_name);
    }

    err_fread_noeof(&first_base, sizeof(uint8_t), 1, saStream);
    err_fread_noeof(&sentinel_index, sizeof(int64_t), 1, saStream);

    // load suffix array
#ifdef SA_COMPRESSION
    int64_t reference_seq_len_ = (reference_seq_len >> SA_COMPX) + 1;
    sa_ms_byte = (int8_t *)_mm_malloc(reference_seq_len_ * sizeof(int8_t), 64);
    sa_ls_word = (uint32_t *)_mm_malloc(reference_seq_len_ * sizeof(uint32_t), 64);
    err_fread_noeof(sa_ms_byte, sizeof(int8_t), reference_seq_len_, saStream);
    err_fread_noeof(sa_ls_word, sizeof(uint32_t), reference_seq_len_, saStream);
#else
    sa_ms_byte = (int8_t *)_mm_malloc(reference_seq_len * sizeof(int8_t), 64);
    sa_ls_word = (uint32_t *)_mm_malloc(reference_seq_len * sizeof(uint32_t), 64);
    err_fread_noeof(sa_ms_byte, sizeof(int8_t), reference_seq_len, saStream);
    err_fread_noeof(sa_ls_word, sizeof(uint32_t), reference_seq_len, saStream);
#endif

    // Load the packed BWT
    strcpy_s(index_file_name, PATH_MAX, ref_file_name);
    strcat_s(index_file_name, PATH_MAX, ".packed.bwt");
    packedBwtStream = fopen(index_file_name,"rb");
    if (packedBwtStream == NULL)
    {
        fprintf(stderr, "ERROR! Unable to open the file: %s\n", index_file_name);
        exit(EXIT_FAILURE);
    }
    else
    {
        fprintf(stderr, "* Index file found. Loading index from %s\n", index_file_name);
    }

    int64_t ref_seq_len_aligned = ((reference_seq_len + CP_BLOCK_SIZE - 1) / CP_BLOCK_SIZE) * CP_BLOCK_SIZE;
    int64_t packedBwtSize = ref_seq_len_aligned / 4 * sizeof(uint8_t);
    packed_bwt = (uint8_t*)malloc(packedBwtSize);
    err_fread_noeof(packed_bwt, packedBwtSize, 1, packedBwtStream);
    fclose(packedBwtStream);
    fclose(saStream);

    int64_t ii = 0;
    for(ii = 0; ii < 5; ii++)// update read count structure
    {
        count[ii] = count[ii] + 1;
    }
    for(ii = 0; ii < 17; ii++)// update read count structure
    {
        count2[ii] = count2[ii] + 1;
    }
    for (ii = 4 * (3 - (int64_t)first_base); ii < 17; ii++)
    {
        count2[ii] = count2[ii] + 1;
    }

}

