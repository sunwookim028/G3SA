#ifndef PIPELINE_H
#define PIPELINE_H

#include <vector>
#include <string>
#include <queue>
#include "concurrentqueue.h"
#include "gpu_types.h"
#include "macro.h"


typedef struct
{
	mem_opt_t *opt;
	mem_pestat_t *pes0;
    long load_chunk_bytes;
	bwaidx_t *idx;
	kmers_bucket_t *kmerHashTab;
    fmindex_t loadedIndex;
    std::ostream *samout;
    g3_opt_t *g3_opt;
    int load_thread_cnt;
    int dispatch_thread_cnt;
    int fd_input;
    process_data_t *proc[MAX_NUM_GPUS][2]; // [gpu][buf]: double-buffer per GPU
} pipeline_aux_t;

typedef struct {
    long long chunk_offset;
    int chunk_size;
    std::vector<int> seq_offsets;
    std::vector<uint8_t> seq;

    std::vector<int> name_offsets;
    std::string name;
    std::vector<int> qual_offsets;
    std::string qual;
} parsed_chunk_t;


typedef struct {
    long long chunk_offset;
    int chunk_size;
    std::vector<int> seq_offsets;
    std::vector<uint8_t> seq;
    std::vector<int> name_offsets;
    std::string name;
    std::vector<int> qual_offsets;
    std::string qual;

    std::vector<int> offsets;
    std::vector<int> rids;
    std::vector<int64_t> positions;
    std::vector<int> ncigars;
    std::vector<uint32_t> cigars;
    std::vector<int> flags;
    std::vector<int> mapqs;


    std::string sambuf;
} aligned_chunk_t;


// top level pipeline wrapper.
void pipeline(pipeline_aux_t *aux);


// thread_cnt threads read then parse fastq inputs from file
// fd_input load_chunk_bytes each from the start (boundaries are checked)
// and enqueue them to the dispatch_queue.
void worker_load_and_parse(
        g3_opt_t *g3_opt,
        int fd_input,
        long load_chunk_bytes,
        int tid,
        int thread_cnt,
        moodycamel::ConcurrentQueue<parsed_chunk_t *> &dispatch_queue
        );


struct writequeue_compare {
    bool operator()(const aligned_chunk_t *a, const aligned_chunk_t *b) const {
        return a->chunk_offset > b->chunk_offset; // in-order
    }
};

void worker_dispatch(
        moodycamel::ConcurrentQueue<parsed_chunk_t *> &dispatch_queue,
        int tid,
        pipeline_aux_t *aux,
        process_data_t *proc0,
        process_data_t *proc1,
        std::priority_queue<
            aligned_chunk_t *, std::vector<aligned_chunk_t *>, writequeue_compare
        > &write_queue
        );

void worker_samgen(
        int gpuid,
        pipeline_aux_t *aux,
        std::priority_queue<
            aligned_chunk_t *, std::vector<aligned_chunk_t *>, writequeue_compare
        > &write_queue
        );

void worker_write(
        std::priority_queue<
            aligned_chunk_t *, std::vector<aligned_chunk_t *>, writequeue_compare
        > &write_queue,
        std::ostream *samout,
        g3_opt_t *g3_opt
        );

#endif
