#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H
#include <stdint.h>
#include "hash_kmer_index.h"
#include "gpu_types.h"
#include "pipeline.h"

#ifdef __cplusplus
extern "C"{
#endif

    /* initialize a new instance of process_data_t 
        initialize and transfer constant memory on device:
            - user-defined options 
            - index 
            - memory management (no transfer)
        initialize pinned memory for reads on host
        initialize memory for reads on device
        initialize intermediate processing memory on device
        initialize a cuda stream for processing
     */
	process_data_t * device_alloc(
            int gpu_no,
            pipeline_aux_t *aux);

    void memcpy_index(
            process_data_t *instance,
            int gpuid, 
            pipeline_aux_t *aux
            );

    // check if requested # of GPUs are available, exits with 1 if not.
    void check_device_count(int num_requested_gpus);

    // destruct cuda stream.
    void destruct_proc(process_data_t *proc);

    // share read-only index pointers (double-buffer: buf1 re-uses buf0's index).
    void share_index(process_data_t *dst, const process_data_t *src);

    // return free VRAM on the current device (bytes).
    size_t get_free_vram(void);

    void cuda_wrapper_test();

// copies batch_size seqs from host (seq, seq_offset) to device memory (proc).
void memcpy_input(int batch_size, process_data_t *proc,
        uint8_t *seq, int *seq_offset);

void memcpy_output(
        aligned_chunk_t *ac,
        process_data_t * proc);

// async D2H to per-slot pinned host arrays.
// o_stream: offsets-only stream (kept empty; stream-specific sync avoids device-wide stall).
// d2h_stream: bulk data stream (overlaps with next batch's GPU compute).
// Call cuda_d2h_event_record(d2h_done, d2h_stream) after this, then cuda_d2h_event_sync
// in worker_samgen before reading h_out_* arrays.
void memcpy_output_async(
        aligned_chunk_t *ac,
        process_data_t  *proc,
        void            *o_stream_opaque,
        void            *d2h_stream_opaque);

// opaque CUDA stream/event wrappers (callable from g++-compiled pipeline.cpp).
void *cuda_d2h_stream_create(int gpuid);
void  cuda_d2h_stream_destroy(void *s);
void *cuda_o_stream_create(int gpuid);    // lightweight offsets-only stream
void  cuda_o_stream_destroy(void *s);
void *cuda_d2h_event_create(void);
void  cuda_d2h_event_destroy(void *e);
void  cuda_d2h_event_record(void *e, void *s);
void  cuda_d2h_event_sync(void *e);
void  cuda_set_device(int gpuid);


#ifdef __cplusplus
}
#endif

#endif
