/*
 * Portions of this file are adapted from minhhpham/bwa
 * (https://github.com/minhhpham/bwa), Copyright (c) minhhpham,
 * GPL-3.0 licensed — a GPU port of BWA (Copyright (c) Genome Research Ltd.,
 * Broad Institute, Dana-Farber Cancer Institute). This file is part of
 * G3SA and is licensed under GPL-3.0 (see LICENSE).
 */
#include "gmem_alloc.cuh"
#include "errchk.cuh"
#include <iostream>
#include <cstdlib>
#define NBUFFERPOOLS 64 	// number of buffer pools (0-31: seeding/chaining; 32-63: extension/traceback)

long long each_pool_size;

// ── Pool profiling (gated by G3_POOL_PROFILE=1) ──────────────────────────
static bool   g_pool_profile = false;
static long long g_peak_offset[NBUFFERPOOLS] = {0};

__device__ unsigned cudaKernelSizeOf(void* ptr){
	/* return the size of the memory chunk that starts with ptr */
	return *(unsigned*)((char*)ptr - 4);
}

__host__ void* CUDA_BufferInit(long long buffer_capacity){
	g_pool_profile = (getenv("G3_POOL_PROFILE") != nullptr);
	/*
	   Allocate NBUFFERPOOLS Buffer pools.
	   First few bytes of each pool contain cuda_mem_info_t
	   return array of pointers to the pools
	 */
	// allocate array of pointers on host
	void** pools;
	each_pool_size = buffer_capacity / NBUFFERPOOLS;
	pools = (void**)malloc(NBUFFERPOOLS*sizeof(void*));

	// allocate NBUFFERPOOLS on device
	for (int i=0; i<NBUFFERPOOLS; i++)
		CUDA_CHECK(cudaMalloc(&pools[i], each_pool_size));

	// allocate array of pointers on device and copy the pool pointers over
	void** d_pools;
	CUDA_CHECK(cudaMalloc((void**)&d_pools, NBUFFERPOOLS*sizeof(void*)));
	if(d_pools == nullptr){
		std::cerr << "* d_pools malloc failed.\n";
		exit(1);
	}
	CUDA_CHECK(cudaMemcpy(d_pools, pools, NBUFFERPOOLS*sizeof(void*), cudaMemcpyHostToDevice));

	free(pools);
	return (void*)d_pools;
}

__host__ void CUDAResetBufferPool(void* d_buffer_pools)
{
	// first coppy the array of pool pointers to host
	void** h_pools;
	h_pools = (void**)malloc(NBUFFERPOOLS*sizeof(void*));
	CUDA_CHECK(cudaMemcpy(h_pools, (void**)d_buffer_pools, NBUFFERPOOLS*sizeof(void*), cudaMemcpyDeviceToHost));

	// reset memory info at the head of each pool and zero all pool memory
	cuda_mem_info_t d_pool_info;		// intermediate data on host
	for (int i = 0; i < NBUFFERPOOLS; i++){
		// find address of the start of the pool
		void* pool_addr = ((void**)h_pools)[i];
		// set base offset
		d_pool_info.current_offset = sizeof(cuda_mem_info_t);
		// set limit of the pool
		d_pool_info.end_offset = (unsigned)(each_pool_size);
		// copy d_pool_info to the start of the pool
		CUDA_CHECK(cudaMemcpy(pool_addr, &d_pool_info, sizeof(cuda_mem_info_t), cudaMemcpyHostToDevice));
	}

	free(h_pools);
}

// Kernel: reset all NBUFFERPOOLS pool headers on the GPU.
// Each thread i resets pool i's cuda_mem_info_t header directly.
static __global__ void reset_pool_headers_kernel(void** d_pools,
                                                  unsigned init_offset,
                                                  unsigned end_offset)
{
    int i = threadIdx.x;
    if (i >= NBUFFERPOOLS) return;
    cuda_mem_info_t *info = (cuda_mem_info_t*)d_pools[i];
    info->current_offset = init_offset;
    info->end_offset     = end_offset;
}

__host__ void CUDAResetBufferPoolAsync(void* d_buffer_pools, cudaStream_t stream)
{
    unsigned init_offset = (unsigned)sizeof(cuda_mem_info_t);
    unsigned end_off     = (unsigned)each_pool_size;
    reset_pool_headers_kernel<<<1, NBUFFERPOOLS, 0, stream>>>(
            (void**)d_buffer_pools, init_offset, end_off);
}

// Kernel: reset a contiguous range [pool_start, pool_end) of pool headers on the GPU.
static __global__ void reset_pool_range_kernel(void** d_pools,
                                               int pool_start,
                                               int pool_end,
                                               unsigned init_offset,
                                               unsigned end_offset)
{
    int i = pool_start + threadIdx.x;
    if (i >= pool_end) return;
    cuda_mem_info_t *info = (cuda_mem_info_t*)d_pools[i];
    info->current_offset = init_offset;
    info->end_offset     = end_offset;
}

// Reset pools [pool_start, pool_end) asynchronously on the given stream.
// Use this between pipeline stages to reclaim bump-allocated memory
// (e.g., reset seeding/chaining pools 0-31 after extension completes).
__host__ void CUDAResetBufferPoolRange(void* d_buffer_pools,
                                       int pool_start, int pool_end,
                                       cudaStream_t stream)
{
    int n = pool_end - pool_start;
    if (n <= 0) return;
    unsigned init_offset = (unsigned)sizeof(cuda_mem_info_t);
    unsigned end_off     = (unsigned)each_pool_size;
    reset_pool_range_kernel<<<1, n, 0, stream>>>(
            (void**)d_buffer_pools, pool_start, pool_end,
            init_offset, end_off);
}

__device__ void* CUDAKernelSelectPool(void* d_buffer_pools, int i){
	/* return pointer to the selected buffer pool */
	return ((void**)d_buffer_pools)[i];
}

__device__ void* CUDAKernelMalloc(void* d_buffer_pool, size_t size, uint8_t align_size){
	/* Malloc function to be run within kernel 
	   return pointer to a chunk of global memory
d_buffer_pool: pointer to a chunk in global memory that was allocated by CUDA_BufferInit
align_size: size of alignment of the chunk. The returned pointer is divisible by align_size (expect 1, 2, 4, 8, power of 2)
The 4 bytes before the returned pointer is the size of the chunk
	 */
	if (size==0) return 0;
	cuda_mem_info_t* d_pool_info = (cuda_mem_info_t*)d_buffer_pool;
	unsigned offset = atomicAdd(&d_pool_info->current_offset, 3+4+(align_size-1)+size); // 3+4 is padding for size + its alignment

	// enforce memory alignment
	// size pointer need to be divisible by 4
	if (offset%4)
		offset += 4 - (offset%4);
	// out pointer need to be divisible by align_size
	if ((offset+4)%align_size)
		offset += align_size - (offset+4)%align_size;

	// check if we passed the end pointer
	if (offset > d_pool_info->end_offset){
		printf("FATAL ERROR: Buffer OOM at blockID %d threadID %d\n", blockIdx.x, threadIdx.x);
		__trap();
	}
	// store size info in first 4 bytes
	unsigned* size_ptr = (unsigned*)((char*)d_buffer_pool + offset);
	*size_ptr = (unsigned)size;
	// output pointer
	void* out_ptr = (void*)((char*)d_buffer_pool + offset + 4);
	return out_ptr;
}

__device__ void* CUDAKernelCalloc(void* d_buffer_pool, size_t num, size_t size, uint8_t align_size){
	/* Calloc function to be run within kernel 
	   allocate num blocks, each with size "size"
	   return pointer to first block
d_buffer_pool: pointer to a chunk in global memory that was allocated by CUDA_BufferInit
	 */
	if (size==0) return 0;
	void* outptr = CUDAKernelMalloc(d_buffer_pool, num*size, align_size);

	memset(outptr, 0, num*size);

	return outptr;
}

__device__ void* CUDAKernelRealloc(void* d_buffer_pool, void* d_current_ptr, size_t new_size, uint8_t align_size){
	/* Realloc function to be run within kernel.
d_buffer_pool: pointer to a chunk in global memory that was allocated by CUDA_BufferInit. If this is null, only do malloc
d_current_ptr: pointer to current memory block
old_size: size of current 
if new_size<old_size, simply change the size value *(d_current_ptr-4)
otherwise, allocate a bigger chunk and copy over
	 */
	if (d_current_ptr == 0){
		return CUDAKernelMalloc(d_buffer_pool, new_size, align_size);
	}

	unsigned old_size = *(unsigned*)((char*)d_current_ptr - 4);

	if (old_size < new_size){
		void* out_ptr	= CUDAKernelMalloc(d_buffer_pool, new_size, align_size);
		cudaKernelMemcpy(d_current_ptr, out_ptr, old_size);
		return out_ptr;
	} else {
		unsigned* size_ptr = (unsigned*)((char*)d_current_ptr - 4);
		*size_ptr = new_size;
		return d_current_ptr;
	}
}

__device__ void cudaKernelMemcpy(void* from, void* to, size_t len){
	/* a memcpy function that can be called within cuda kernel */
	memcpy(to, from, len);
}

__device__ void cudaKernelMemmove(void* from, void* to, size_t len){
	/* a memmove function that can be called within cuda kernel */
	int i;	// byte counter
	if (from < to){
		// reverse copy
		for (i = len-1; i >= 0; i-=sizeof(char))
			((char*)to)[i] = ((char*)from)[i];
	} else {
		// forward copy
		for (i = 0; i < len; i+=sizeof(char))
			((char*)to)[i] = ((char*)from)[i];
	}	
}

// Capture peak current_offset for pools [start, end) — call before any reset of that range.
// Syncs the stream, so only call when G3_POOL_PROFILE=1 (gated in callers).
__host__ void capturePoolPeak(void* d_buffer_pools, int start, int end, cudaStream_t stream) {
	if (!g_pool_profile) return;
	cudaStreamSynchronize(stream);
	void** h_pools = (void**)malloc(NBUFFERPOOLS * sizeof(void*));
	cudaMemcpy(h_pools, d_buffer_pools, NBUFFERPOOLS * sizeof(void*), cudaMemcpyDeviceToHost);
	for (int i = start; i < end; i++) {
		cuda_mem_info_t info;
		cudaMemcpy(&info, h_pools[i], sizeof(cuda_mem_info_t), cudaMemcpyDeviceToHost);
		if ((long long)info.current_offset > g_peak_offset[i])
			g_peak_offset[i] = info.current_offset;
	}
	free(h_pools);
}

// Print per-group peak usage and recommended -M.  Call once after all batches.
__host__ void printPoolPeaks(void) {
	if (!g_pool_profile) return;
	double pool_mb = each_pool_size / 1048576.0;
	double total_mb = NBUFFERPOOLS * pool_mb;
	fprintf(stderr, "[pool profile] pools=%d  each=%.2f MB  total=%.0f MB\n",
	        NBUFFERPOOLS, pool_mb, total_mb);
	// seeding/chaining group (pools 0-31)
	long long peak_sc = 0;
	for (int i = 0; i < 32; i++) if (g_peak_offset[i] > peak_sc) peak_sc = g_peak_offset[i];
	fprintf(stderr, "[pool profile] seeding/chain (0-31): peak %.2f MB / %.2f MB  (%.1f%%)\n",
	        peak_sc / 1048576.0, pool_mb, 100.0 * peak_sc / each_pool_size);
	// extension/traceback group (pools 32-63)
	long long peak_et = 0;
	for (int i = 32; i < 64; i++) if (g_peak_offset[i] > peak_et) peak_et = g_peak_offset[i];
	fprintf(stderr, "[pool profile] ext/traceback (32-63): peak %.2f MB / %.2f MB  (%.1f%%)\n",
	        peak_et / 1048576.0, pool_mb, 100.0 * peak_et / each_pool_size);
	// recommended -M: total demand = 64 pools * worst-group peak, with 2x headroom
	long long worst = peak_sc > peak_et ? peak_sc : peak_et;
	long long rec_mb = (long long)(NBUFFERPOOLS * worst / 1048576.0 * 2.0 + 0.5);
	fprintf(stderr, "[pool profile] recommended -M: %lld MB  (current: %.0f MB)\n",
	        rec_mb, total_mb);
}

/* Optional debugging utility (not called internally): prints per-pool usage
 * fractions to stderr from the host. Kept as a public API for callers that
 * want ad hoc pool-usage introspection. */
__global__ void printBufferInfoHost_kernel(void* d_buffer_pools, float* d_usage){
	int i = threadIdx.x;
	void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, i);
	cuda_mem_info_t* d_pool_info = (cuda_mem_info_t*)d_buffer_ptr;
	d_usage[i] =  (float)d_pool_info->current_offset/d_pool_info->end_offset;			
}

void printBufferInfoHost(void* d_buffer_pools){
	float *h_usage, *d_usage;
	h_usage = (float*)malloc(NBUFFERPOOLS*sizeof(float));
	cudaMalloc((void**)&d_usage, NBUFFERPOOLS*sizeof(float));
	printBufferInfoHost_kernel <<< 1, NBUFFERPOOLS >>> (d_buffer_pools, d_usage);
	cudaMemcpy(h_usage, d_usage, NBUFFERPOOLS*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i=0; i<NBUFFERPOOLS; i++)
		fprintf(stderr, "pool %2d: %.2f used\n", i, h_usage[i]);
}
