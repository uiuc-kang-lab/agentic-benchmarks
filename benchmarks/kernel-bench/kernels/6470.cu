#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <type_traits>

#define BLOCK_SIZE 256

// Kernel that uses stride loops to sum up elements, handling workloads larger than the number of available threads
template <typename scalar_t>
__global__ void mean_reduce_kernel_stride(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t L,  // length of the reduction dimension
    int64_t stride,  // stride in the input tensor corresponding to the reduction dimension position
    int64_t N  // total number of output elements
) {
    int out_idx = blockIdx.x;  // Each block handles one output element
    if (out_idx >= N) return;

    // Calculate base offset along the input tensor for the current output element
    int64_t base_offset = (out_idx / stride) * L * stride + (out_idx % stride);

    scalar_t sum = 0;

    // Use stride loop to handle dimensional reduction larger than the thread block
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        sum += input[base_offset + i * stride];
    }

    // Use shared memory to perform the reduction within the block
    __shared__ scalar_t sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Parallel reduction within the block
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The first thread writes the result back to global memory
    if (threadIdx.x == 0) {
        output[out_idx] = sdata[0] / static_cast<scalar_t>(L);
    }
}

// Host function to configure and launch the kernel
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    // Get input sizes and calculate reduction dimension size, outer and inner sizes
    auto sizes = input.sizes();
    int64_t L = sizes[dim];
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    int64_t N = outer_size * inner_size;  // Total number of output elements

    // Define the reduced tensor shape and create output tensor
    std::vector<int64_t> output_sizes(sizes.begin(), sizes.end());
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Launch kernel
    const int threads = BLOCK_SIZE;
    const int blocks = static_cast<int>(N);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda_stride", ([&] {
        mean_reduce_kernel_stride<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            L,
            inner_size,
            N
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean Reduction (Stride Loop CUDA)");
}
