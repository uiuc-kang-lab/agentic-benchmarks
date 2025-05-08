#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void hybrid_mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    extern __shared__ char shared_mem[];
    scalar_t* warp_results = reinterpret_cast<scalar_t*>(shared_mem);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int out_idx = blockIdx.x;

    if (out_idx >= outer_size * inner_size)
        return;

    const int outer_idx = out_idx / inner_size;
    const int inner_idx = out_idx % inner_size;
    const int base_idx = outer_idx * dim_size * inner_size + inner_idx;

    // Each thread processes multiple elements with grid-stride loop
    scalar_t thread_sum = 0;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        thread_sum += input[base_idx + i * inner_size];
    }

    // First level: warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);

    // Store the warp results
    if (lane == 0) {
        warp_results[wid] = thread_sum;
    }
    __syncthreads();

    // Final reduction: first warp reduces all warp results
    if (tid < 32) {
        scalar_t final_sum = 0;
        if (tid < warps_per_block) {
            final_sum = warp_results[tid];
        }
        final_sum = warp_reduce_sum(final_sum);

        if (lane == 0) {
            output[out_idx] = final_sum / static_cast<scalar_t>(dim_size);
        }
    }
}

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    const int threads = 512;
    const int blocks = outer_size * inner_size;
    const int warps_per_block = threads >> 5;
    const int shared_mem_size = warps_per_block * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        hybrid_mean_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean reduction (CUDA)");
}