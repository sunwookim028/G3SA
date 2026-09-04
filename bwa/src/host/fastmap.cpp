/* The MIT License

   Copyright (c) 2018-     Dana-Farber Cancer Institute
                 2009-2018 Broad Institute, Inc.
                 2008-2009 Genome Research Ltd. (GRL)

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
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <cerrno>
#include <ctype.h>
#include <math.h>
#include "macro.h"
#include "timer.h"
#include "fmi_wrapper.h"
#include <time.h>

#include <iostream>

#include "cuda_wrapper.h"

#include "fastmap.h"
#include "utils.h"

#include "khash.h"
KHASH_MAP_INIT_STR(str, int)

#include "pipeline.h"

g3_opt_t *g3_opt_init()
{
    g3_opt_t *o;
    o = (g3_opt_t*)calloc(1, sizeof(g3_opt_t));
    o->num_use_gpus = 1;
    o->baseline = 0;
    o->verbosity = 0;
    o->step_count = 20;
    o->print_mask = 0;
    o->batch_size = 10000; // must be <= MAX_BATCH_SIZE (80000); use -Z to override
    o->bound_load_queue = false;
    o->single_record_bytes = 350; // avg bytes per FASTQ record for load chunk calculation
    o->monitoring_period = 10;
    o->load_thread_factor = 1;     // default: 1 load thread per GPU
    o->dispatch_thread_factor = 1; // default: 1 dispatch thread per GPU
    o->pool_mb = 1000;             // 1 GB default; override with -M for smaller GPUs
    return o;
}

mem_opt_t *mem_opt_init()
{
	mem_opt_t *o;
	o = (mem_opt_t*)calloc(1, sizeof(mem_opt_t));
	o->flag = 0;
	o->a = 1; o->b = 4;
	o->o_del = o->o_ins = 6;
	o->e_del = o->e_ins = 1;
	o->w = 100;
	o->T = 30;
	o->zdrop = 100;
	o->pen_unpaired = 17;
	o->pen_clip5 = o->pen_clip3 = 5;

	o->max_mem_intv = 20;

	o->min_seed_len = 19;
	o->split_width = 10;
	o->max_occ = 500;
	o->max_chain_gap = 10000;
	o->max_ins = 10000;
	o->mask_level = 0.50;
	o->drop_ratio = 0.50;
	o->XA_drop_ratio = 0.80;
	o->split_factor = 1.5;
	o->chunk_size = 30000000;
	o->n_threads = 1;
	o->max_XA_hits = 5;
	o->max_XA_hits_alt = 200;
	o->max_matesw = 50;
	o->mask_level_redun = 0.95;
	o->min_chain_weight = 0;
	o->max_chain_extend = 1<<30;
	o->mapQ_coef_len = 50; o->mapQ_coef_fac = log(o->mapQ_coef_len);
	bwa_fill_scmat(o->a, o->b, o->mat);
	return o;
}

void bwa_fill_scmat(int a, int b, int8_t mat[25])
{
	int i, j, k;
	for (i = k = 0; i < 4; ++i) {
		for (j = 0; j < 4; ++j)
			mat[k++] = i == j? a : -b;
		mat[k++] = -1; // ambiguous base
	}
	for (j = 0; j < 5; ++j) mat[k++] = -1;
}

static void update_a(mem_opt_t *opt, const mem_opt_t *opt0)
{
	if (opt0->a) { // matching score is changed
		if (!opt0->b) opt->b *= opt->a;
		if (!opt0->T) opt->T *= opt->a;
		if (!opt0->o_del) opt->o_del *= opt->a;
		if (!opt0->e_del) opt->e_del *= opt->a;
		if (!opt0->o_ins) opt->o_ins *= opt->a;
		if (!opt0->e_ins) opt->e_ins *= opt->a;
		if (!opt0->zdrop) opt->zdrop *= opt->a;
		if (!opt0->pen_clip5) opt->pen_clip5 *= opt->a;
		if (!opt0->pen_clip3) opt->pen_clip3 *= opt->a;
		if (!opt0->pen_unpaired) opt->pen_unpaired *= opt->a;
	}
}


/*********************
 * Full index reader *
 *********************/

char *bwa_idx_infer_prefix(const char *hint)
{
	char *prefix;
	int l_hint;
	FILE *fp;
	l_hint = strlen(hint);
	prefix = (char*)malloc(l_hint + 3 + 4 + 1);
	strcpy(prefix, hint);
	strcpy(prefix + l_hint, ".64.bwt");
	if ((fp = fopen(prefix, "rb")) != 0) {
		fclose(fp);
		prefix[l_hint + 3] = 0;
		return prefix;
	} else {
		strcpy(prefix + l_hint, ".bwt");
		if ((fp = fopen(prefix, "rb")) == 0) {
			free(prefix);
			return 0;
		} else {
			fclose(fp);
			prefix[l_hint] = 0;
			return prefix;
		}
	}
}

bwt_t *bwa_idx_load_bwt(const char *hint)
{
	char *tmp, *prefix;
	bwt_t *bwt;
	prefix = bwa_idx_infer_prefix(hint);
	if (prefix == 0) {
		fprintf(stderr, "[E::%s] fail to locate the index files\n", __func__);
		return 0;
	}
	tmp = (char*)calloc(strlen(prefix) + 5, 1);
	strcat(strcpy(tmp, prefix), ".bwt"); // FM-index
	bwt = bwt_restore_bwt(tmp);
	strcat(strcpy(tmp, prefix), ".sa");  // partial suffix array (SA)
	bwt_restore_sa(tmp, bwt);
	free(tmp); free(prefix);
	return bwt;
}

bwaidx_t *bwa_idx_load_from_disk(const char *hint, int which)
{
	bwaidx_t *idx;
	char *prefix;
	prefix = bwa_idx_infer_prefix(hint);
	if (prefix == 0) {
        fprintf(stderr, "hint: %s\n", hint);

		fprintf(stderr, "[E::%s] fail to locate the index files\n", __func__);
		return 0;
	}
	idx = (bwaidx_t*)calloc(1, sizeof(bwaidx_t));
	if (which & BWA_IDX_BWT) idx->bwt = bwa_idx_load_bwt(hint);
	if (which & BWA_IDX_BNS) {
		int i, c;
		idx->bns = bns_restore(prefix);
        if (which & BWA_IDX_PAC) {
			idx->pac = (uint8_t*)calloc(idx->bns->l_pac/4+1, 1);
			fread(idx->pac, 1, idx->bns->l_pac/4+1, idx->bns->fp_pac); // concatenated 2-bit encoded sequence
			fclose(idx->bns->fp_pac);
			idx->bns->fp_pac = 0;
		}
	}
	free(prefix);
	return idx;
}

bwaidx_t *bwa_idx_load(const char *hint, int which)
{
	return bwa_idx_load_from_disk(hint, which);
}

void bwa_idx_destroy(bwaidx_t *idx)
{
	if (idx == 0) return;
	if (idx->mem == 0) {
		if (idx->bwt) bwt_destroy(idx->bwt);
		if (idx->bns) bns_destroy(idx->bns);
		if (idx->pac) free(idx->pac);
	} else {
		free(idx->bwt); free(idx->bns->anns); free(idx->bns);
		if (!idx->is_shm) free(idx->mem);
	}
	free(idx);
}


static char *bwa_escape(char *s)
{
	char *p, *q;
	for (p = q = s; *p; ++p) {
		if (*p == '\\') {
			++p;
			if (*p == 't') *q++ = '\t';
			else if (*p == 'n') *q++ = '\n';
			else if (*p == 'r') *q++ = '\r';
			else if (*p == '\\') *q++ = '\\';
		} else *q++ = *p;
	}
	*q = '\0';
	return s;
}

char *bwa_insert_header(const char *s, char *hdr)
{
	int len = 0;
	if (s == 0 || s[0] != '@') return hdr;
	if (hdr) {
		len = strlen(hdr);
		hdr = (char*)realloc(hdr, len + strlen(s) + 2);
		hdr[len++] = '\n';
		strcpy(hdr + len, s);
	} else hdr = strdup(s);
	bwa_escape(hdr + len);
	return hdr;
}


int main_mem(int argc, char *argv[])
{
    std::thread worker_wake_cuda(cuda_wrapper_test);
	mem_opt_t *opt, opt0;
	int fd, c;
	pipeline_aux_t aux;

    struct timespec start, end;
    double walltime;

	memset(&aux, 0, sizeof(pipeline_aux_t));
    std::ofstream samfile;
    std::string samfilepath;
    aux.samout = &std::cout;


    int load_nt_factor = 1;
    int dispatch_nt_factor = 1;
    g3_opt_t *g3_opt = g3_opt_init();
	aux.opt = opt = mem_opt_init();
	memset(&opt0, 0, sizeof(mem_opt_t));
	while((c = getopt(argc, argv, "bL:v:g:i:z:l:p:Z:F:m:M:o:")) >= 0){
		if (c == 'b') g3_opt->baseline = 1;
		else if (c == 'L') g3_opt->bound_load_queue = true;
		else if (c == 'v') g3_opt->verbosity = atoi(optarg);
		else if (c == 'g'){
            g3_opt->num_use_gpus = atoi(optarg);
        }
		else if (c == 'i'){
            load_nt_factor = atoi(optarg);
            aux.load_thread_cnt = load_nt_factor * g3_opt->num_use_gpus;
        }
		else if (c == 'z'){
            dispatch_nt_factor = atoi(optarg);
            aux.dispatch_thread_cnt = dispatch_nt_factor * g3_opt->num_use_gpus;
        }
		else if (c == 'l'){
            g3_opt->step_count = atoi(optarg);
        } 
		else if (c == 'p'){
            g3_opt->print_mask = atoi(optarg);
        } 
		else if (c == 'Z'){
            g3_opt->batch_size = atoi(optarg);
        } 
		else if (c == 'F') {
            g3_opt->single_record_bytes = atoi(optarg); 
        }
		else if (c == 'm') {
            g3_opt->monitoring_period = atoi(optarg);
        }
		else if (c == 'M') {
            g3_opt->pool_mb = atol(optarg);
        }
        else if (c == 'o' || c == 'f') {
            samfilepath = optarg;
            samfile.open(samfilepath);
            if(!samfile.is_open()){
                std::cerr << "ERROR. Could not open output filepath "
                    << samfilepath << std::endl;
                exit(EXIT_FAILURE);
            }
            aux.samout = &samfile;
        }
		else return 1;
	}
    aux.g3_opt = g3_opt;

    // Set default thread counts if not specified via -i/-z
    if (aux.load_thread_cnt == 0) {
        aux.load_thread_cnt = g3_opt->load_thread_factor * g3_opt->num_use_gpus;
    }
    if (aux.dispatch_thread_cnt == 0) {
        aux.dispatch_thread_cnt = g3_opt->dispatch_thread_factor * g3_opt->num_use_gpus;
    }

    // storage loading batch size
    aux.load_chunk_bytes = g3_opt->batch_size * g3_opt->single_record_bytes;

	bwa_fill_scmat(opt->a, opt->b, opt->mat);

	// load the reference index
    std::cerr << "-------------------------------------------------\n";
    std::cerr << "* LOADING THE INDEX\n";

    // load the large (occ2) index concurrently.
    // Capture ref filename by value before optind is incremented below.
    const char *ref_filename = argv[optind];
    FMI_wrapper *obj;
    std::thread t_load_large_index([&, ref_filename]() {
            obj = FMI_wrapper_create(ref_filename);
            FMI_wrapper_load_index(obj, &(aux.loadedIndex));
    });

    // load other index structures.
    if ((aux.idx = bwa_idx_load(argv[optind], BWA_IDX_BNS|BWA_IDX_PAC)) == 0) return 1;
	aux.kmerHashTab = loadKMerIndex(argv[optind++]);

    // open the input file
    fd = open(argv[optind], O_RDONLY);
    if (fd < 0) {
        std::cerr << "* Error opening file: " << argv[optind] << "\n";
        return 1;
    }
    aux.fd_input = fd;
    std::cerr << "-------------------------------------------------\n";
    std::cerr << "* INPUT FILE OPENED\n";

    // allocate device buffers.
    worker_wake_cuda.join(); // verify cuda is initialized.
    std::cerr << "-------------------------------------------------\n";
    std::cerr << "* ALLOCATING DEVICE BUFFERS\n";
    check_device_count(aux.g3_opt->num_use_gpus);
    for(int j=0; j<aux.g3_opt->num_use_gpus; j++){
        aux.proc[j][0] = device_alloc(j, &aux);
        aux.proc[j][1] = nullptr; // allocated after index load (see pipeline.cpp gpu_worker)
    }

    t_load_large_index.join(); // sync

    clock_gettime(CLOCK_REALTIME,  &start);
    pipeline(&aux);     // Main processing
    clock_gettime(CLOCK_REALTIME,  &end);
    walltime = (end.tv_sec - start.tv_sec) +\
                           (end.tv_nsec - start.tv_nsec) / 1e9;
    std::cerr << "-------------------------------------------------\n";
    std::cerr << "* RUNTIME BREAKDOWN\n";
    std::cerr << "* Wall-clock time after loading the index: "
        << walltime << " seconds" << std::endl;


    report_stats(tprof, g3_opt);

    std::cerr << "-------------------------------------------------\n";
    std::cerr << "* HYPERPARAMETERS\n";
    std::cerr << "* per GPU batch size (count) = " << g3_opt->batch_size << "\n";
    std::cerr << "* storage loading chunk size = " << aux.load_chunk_bytes / MB_SIZE<< " MB\n";
    std::cerr << "  (assumed " << g3_opt->single_record_bytes
        << " bytes per a single input record)\n";
    std::cerr << "* SPAWNed " << load_nt_factor
        << " file loading threads per GPU\n";
    std::cerr << "* SPAWNed " << dispatch_nt_factor
        << " job dispatching threads per GPU\n";
    std::cerr << "-------------------------------------------------\n";

	FMI_wrapper_destroy(obj);
	free(opt);
    free(g3_opt);
	bwa_idx_destroy(aux.idx);
	return 0;
}
