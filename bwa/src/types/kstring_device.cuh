#ifndef KSTRING_DEVICE_CUH
#define KSTRING_DEVICE_CUH

// Device-side kstring_t; see kstring_device.cu for attribution (adapted from
// minhhpham/bwa's cuda/kstring_CUDA.cu).

typedef struct __kstring_t {
	size_t l, m;
	char *s;
} kstring_t;

extern __device__ void ks_resize(kstring_t *s, size_t size, void* d_buffer_ptr);
extern __device__ int kputsn(const char *p, int l, kstring_t *s, void* d_buffer_ptr);
extern __device__ int kputc(int c, kstring_t *s, void* d_buffer_ptr);
extern __device__ int kputw(int c, kstring_t *s, void* d_buffer_ptr);
extern __device__ int kputl(long c, kstring_t *s, void* d_buffer_ptr);
#endif
