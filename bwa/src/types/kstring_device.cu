/*
 * Portions of this file are adapted from minhhpham/bwa
 * (https://github.com/minhhpham/bwa), Copyright (c) minhhpham,
 * GPL-3.0 licensed, a GPU port of BWA (Copyright (c) Genome Research Ltd.,
 * Broad Institute, Dana-Farber Cancer Institute). This file is part of
 * G3SA and is licensed under GPL-3.0 (see LICENSE).
 */

#include "gmem_alloc.cuh"
#include "kstring_device.cuh"
#include <stdio.h>
#include <stdarg.h>

#define kroundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))

__device__ void ks_resize(kstring_t *s, size_t size, void* d_buffer_ptr)
{
	if (s->m < size) {
		s->m = size;
		kroundup32(s->m);
		s->s = (char*)CUDAKernelRealloc(d_buffer_ptr, s->s, s->m, 1);
	}
}

/* concatenate string p to the end of string s->s */
__device__ int kputsn(const char *p, int l, kstring_t *s, void* d_buffer_ptr)
{
	if (s->l + l + 1 >= s->m) {
		s->m = s->l + l + 2;
		kroundup32(s->m);
		s->s = (char*)CUDAKernelRealloc(d_buffer_ptr, s->s, s->m, 1);
	}
	cudaKernelMemcpy((void*)p, s->s + s->l, l);
	s->l += l;
	s->s[s->l] = 0;
	return l;
}

/* add one char (c) to string s->s*/
__device__ int kputc(int c, kstring_t *s, void* d_buffer_ptr)
{
	if (s->l + 1 >= s->m) {
		s->m = s->l + 2;
		kroundup32(s->m);
		s->s = (char*)CUDAKernelRealloc(d_buffer_ptr, s->s, s->m, 1);
	}
	s->s[s->l++] = c;
	s->s[s->l] = 0;
	return c;
}

__device__ int kputw(int c, kstring_t *s, void* d_buffer_ptr)
{
	char buf[16];
	int l, x;
	if (c == 0) return kputc('0', s, d_buffer_ptr);
	for (l = 0, x = c < 0? -c : c; x > 0; x /= 10) buf[l++] = x%10 + '0';
	if (c < 0) buf[l++] = '-';
	if (s->l + l + 1 >= s->m) {
		s->m = s->l + l + 2;
		kroundup32(s->m);
		s->s = (char*)CUDAKernelRealloc(d_buffer_ptr, s->s, s->m, 1);
	}
	for (x = l - 1; x >= 0; --x) s->s[s->l++] = buf[x];
	s->s[s->l] = 0;
	return 0;
}

__device__ int kputl(long c, kstring_t *s, void* d_buffer_ptr)
{
	char buf[32];
	long l, x;
	if (c == 0) return kputc('0', s, d_buffer_ptr);
	for (l = 0, x = c < 0? -c : c; x > 0; x /= 10) buf[l++] = x%10 + '0';
	if (c < 0) buf[l++] = '-';
	if (s->l + l + 1 >= s->m) {
		s->m = s->l + l + 2;
		kroundup32(s->m);
		s->s = (char*)CUDAKernelRealloc(d_buffer_ptr, s->s, s->m, 1);
	}
	for (x = l - 1; x >= 0; --x) s->s[s->l++] = buf[x];
	s->s[s->l] = 0;
	return 0;
}
