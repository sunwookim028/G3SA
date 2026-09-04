/*
 * Portions of this file are adapted from minhhpham/bwa
 * (https://github.com/minhhpham/bwa), Copyright (c) minhhpham,
 * GPL-3.0 licensed — a GPU port of BWA (Copyright (c) Genome Research Ltd.,
 * Broad Institute, Dana-Farber Cancer Institute). This file is part of
 * G3SA and is licensed under GPL-3.0 (see LICENSE).
 */
#ifndef GMEM_ALLOC_CUH
#define GMEM_ALLOC_CUH
#include <stdio.h>
#include <stdint.h>

typedef struct
{
	unsigned current_offset;	// current offset to the available part of the chunk
	unsigned end_offset;		// the max offset of the chunk
} cuda_mem_info_t;



extern __host__ void* CUDA_BufferInit(long long buffer_capacity);
// reset buffer pools (synchronous — kept for init/teardown)
extern __host__ void CUDAResetBufferPool(void* big_pool);
// reset buffer pools asynchronously on the given stream
extern __host__ void CUDAResetBufferPoolAsync(void* big_pool, cudaStream_t stream);
// reset a contiguous range [pool_start, pool_end) of pools asynchronously (stage-level arena reset)
extern __host__ void CUDAResetBufferPoolRange(void* big_pool, int pool_start, int pool_end, cudaStream_t stream);

/* FUNCTION TO DO MALLOC AND REALLOC WITHIN CUDA KERNELS */
// select a buffer pool from the big pool
extern __device__ void* CUDAKernelSelectPool(void* big_pool, int i);
// malloc within kernel
extern __device__ void* CUDAKernelMalloc(void* d_mem_chunk_ptr, size_t size, uint8_t align_size);
extern __device__ void* CUDAKernelCalloc(void* d_mem_chunk_ptr, size_t num, size_t size, uint8_t align_size);
// realloc within kernel
extern __device__ void* CUDAKernelRealloc(void* d_mem_chunk_ptr, void* d_current_ptr, size_t new_size, uint8_t align_size);
// memcpy within kernel
extern __device__ void cudaKernelMemcpy(void* from, void* to, size_t len);
// memmove within kernel
extern __device__ void cudaKernelMemmove(void* from, void* to, size_t len);
// Optional debugging utility (not called internally); prints pool usage from the host.
extern void printBufferInfoHost(void* d_buffer_pools);
// Pool profiling (enabled by G3_POOL_PROFILE=1 env var)
extern void capturePoolPeak(void* d_buffer_pools, int start, int end, cudaStream_t stream);
extern void printPoolPeaks(void);

// GPU-side vector helpers (kvec-style) using CUDAKernelRealloc.
#define KV_GPU_ALIGN_INT 4
#define KV_GPU_ALIGN_PTR 8
#define KV_GPU_MIN_CAPACITY 2

#define KV_GPU_GROW(type, v, field, buffer, align) do {                 \
        if ((v).n == (v).m) {                                           \
                (v).m = (v).m ? ((v).m << 1) : KV_GPU_MIN_CAPACITY;      \
                (v).field = (type*)CUDAKernelRealloc(                   \
                        (buffer), (v).field, sizeof(type) * (v).m, (align)); \
        }                                                               \
} while (0)

#define KV_GPU_PUSH(type, v, field, x, buffer, align) do {              \
        KV_GPU_GROW(type, v, field, buffer, align);                     \
        (v).field[(v).n++] = (x);                                       \
} while (0)

#define KV_GPU_PUSHP(type, v, field, buffer, align, out_ptr) do {       \
        KV_GPU_GROW(type, v, field, buffer, align);                     \
        (out_ptr) = &((v).field[(v).n++]);                              \
} while (0)
#endif
