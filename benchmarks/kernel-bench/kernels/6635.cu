#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

template <typename scalar_t>
__global__ void optimized_max_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = outer_size * inner_size;
    
    if (idx >= total_elements) return;
    
    const int outer_idx = idx / inner_size;
    const int inner_idx = idx % inner_size;
    
    const int64_t start_idx = outer_idx * dim_size * inner_size + inner_idx;
    
    scalar_t max_vals[WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    scalar_t max_val = input[start_idx + lane * inner_size];

    // Utilize warp-level primitives to minimize divergence and reduce over dim_size
    for (int i = lane; i < dim_size; i += WARP_SIZE) {
        scalar_t val = input[start_idx + i * inner_size];
        max_val = max(max_val, val);
    }

    // Warp level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    // Only the leader of the warp writes back the result
    if (lane == 0) {
        output[idx] = max_val;
    }
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }
    
    const int64_t dim_size = input.size(dim);
    
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        optimized_max_reduce_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA)");
}
