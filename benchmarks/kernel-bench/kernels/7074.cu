#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__global__ void min_reduce_strided_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner,
    const int total_elements) {

    const int warpSize = 32;
    const int stride = blockDim.x * gridDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % warpSize;
    const int wid = tid / warpSize;

    // Grid-stride loop over warps
    for (int w = wid; w < outer * inner; w += stride / warpSize) {
        const int outer_idx = w / inner;
        const int inner_idx = w % inner;
        const int base = outer_idx * (r * inner) + inner_idx;
        
        // Initialize local minimum
        scalar_t local_min = std::numeric_limits<scalar_t>::max();
        
        // Strided loop over reduction dimension
        #pragma unroll 4
        for (int j = lane; j < r; j += warpSize) {
            const int idx = base + j * inner;
            if (idx < total_elements) {
                const scalar_t val = input[idx];
                local_min = val < local_min ? val : local_min;
            }
        }

        // Warp-level reduction using shuffle operations
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            const scalar_t other = __shfl_down_sync(0xffffffff, local_min, offset);
            local_min = other < local_min ? other : local_min;
        }

        // First thread in warp writes result
        if (lane == 0 && w < outer * inner) {
            output[w] = local_min;
        }
    }
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int ndim = input.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    // Calculate dimensions
    int outer = 1;
    for (int i = 0; i < dim; i++) {
        outer *= input.size(i);
    }
    int r = input.size(dim);
    int inner = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner *= input.size(i);
    }

    // Create output tensor
    std::vector<int64_t> output_shape;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            output_shape.push_back(input.size(i));
        }
    }
    auto output = torch::empty(output_shape, input.options());

    // Calculate kernel launch parameters
    const int total_elements = input.numel();
    const int threads_per_block = 256;
    const int max_blocks = 256;
    const int num_warps_needed = (outer * inner + 31) / 32;
    const int blocks = min(max_blocks, (num_warps_needed * 32 + threads_per_block - 1) / threads_per_block);

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_strided_cuda", ([&] {
        min_reduce_strided_kernel<scalar_t><<<blocks, threads_per_block, 0,
            c10::cuda::getCurrentCUDAStream().stream()>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer,
                r,
                inner,
                total_elements);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided min reduction over a specified dimension (CUDA)");
}