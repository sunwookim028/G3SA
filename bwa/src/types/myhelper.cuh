#ifndef _MYHELPER
#define _MYHELPER
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include "gpu_types.h"
#include "macro.h"

struct get_n {
    __host__ __device__
        int operator()(const mem_alnreg_v& x) const { return x.n; }
};

// assuming proc->d_offsets already allocated.
void compute_offsets_on_host(process_data_t* proc, int batch_size, cudaStream_t stream = 0) {
    thrust::device_ptr<mem_alnreg_v>  reg_ptr = thrust::device_pointer_cast(proc->d_regs);
    auto n_iter = thrust::make_transform_iterator(reg_ptr, get_n{});

    thrust::device_ptr<int> offsets_ptr = thrust::device_pointer_cast(proc->d_offsets);

    // exclusive scan [0, n₀, n₀+n₁, …] into d_offsets[0]…d_offsets[batch_size]
    thrust::exclusive_scan(
            thrust::cuda::par.on(stream),
            n_iter,
            n_iter + batch_size + 1, // +1 for the total sum
            offsets_ptr
            );
}

#endif // _MYHELPER
