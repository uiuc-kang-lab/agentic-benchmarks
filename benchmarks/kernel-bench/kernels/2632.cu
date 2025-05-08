#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename scalar_t>
__device__ inline scalar_t relu(scalar_t val) {
    return val > 0 ? val : 0;
}

template <typename scalar_t>
__global__ void relu_vectorized_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_size = 4;

    // Vectorized memory access
    using vec_t = typename std::conditional<
        std::is_same<scalar_t, float>::value,
        float4,
        int4>::type;

    for(int i = tid * vec_size; i < size; i += stride * vec_size) {
        vec_t in_val = *reinterpret_cast<const vec_t*>(&input[i]);
        vec_t out_val;

        #pragma unroll
        for(int j = 0; j < vec_size; j++) {
            reinterpret_cast<scalar_t*>(&out_val)[j] = 
                relu(reinterpret_cast<scalar_t*>(&in_val)[j]);
        }

        *reinterpret_cast<vec_t*>(&output[i]) = out_val;
    }

    // Handle remaining elements
    const int remainder_start = (size / vec_size) * vec_size;
    for(int i = remainder_start + tid; i < size; i += stride) {
        output[i] = relu(input[i]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    // Optimize block size and grid size for better occupancy
    const int vec_size = 4;
    int block_size = 128;  // Reduced block size for potentially better occupancy
    int max_blocks_per_sm;
    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    
    // Calculate theoretical occupancy
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        relu_vectorized_kernel<scalar_t>,
        block_size,
        0);
    
    // Calculate grid size based on device capabilities and data size
    int num_blocks = (size + (block_size * vec_size - 1)) / (block_size * vec_size);
    num_blocks = std::min(num_blocks, max_blocks_per_sm * num_sm);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_vectorized_kernel", [&] {
        relu_vectorized_kernel<scalar_t><<<num_blocks, block_size>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            size
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized ReLU forward (CUDA)");
}