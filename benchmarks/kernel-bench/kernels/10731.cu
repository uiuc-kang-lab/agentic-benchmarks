#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Optimized CUDA kernel using 1D grid indexing: each block processes one row.
// Assumes the tensor is contiguous, of type float, and the cumulative sum is along the last dimension
// with size not exceeding WARP_SIZE.

__global__ void reverse_cumsum_kernel(const float* input, float* output, int row_size) {
    // Each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Calculate the index for the reversed order
    int idx = row * row_size + (row_size - 1 - tid);
    float val = input[idx];

    // Active mask for the entire warp; since blockDim.x == row_size <= WARP_SIZE, all threads are active
    unsigned mask = __activemask();
    
    // Perform an inclusive scan within the warp using warp shuffles
    #pragma unroll
    for (int offset = 1; offset < row_size; offset *= 2) {
        float shfl_val = __shfl_down_sync(mask, val, offset);
        if (tid + offset < row_size) {
            val += shfl_val;
        }
    }
    
    // Write the computed reverse cumulative sum back in the correct order
    output[idx] = val;
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    // Ensure the tensor is contiguous and on CUDA
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    x = x.contiguous();

    // Ensure proper conditions to use our optimized kernel:
    // - Type must be float
    // - Operation must be on the last dimension
    // - Size along that dimension must not exceed WARP_SIZE
    if (x.scalar_type() != at::kFloat || dim != x.dim() - 1 || x.size(dim) > WARP_SIZE) {
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        return cumsum.flip(dim);
    }

    // Prepare output tensor
    auto output = at::empty_like(x);
    int row_size = x.size(dim);
    int rows = x.numel() / row_size;
    
    // Use 1D grid: one block per row, and each block has row_size threads
    dim3 grid(rows);
    dim3 block(row_size);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    reverse_cumsum_kernel<<<grid, block, 0, stream>>>(input_ptr, output_ptr, row_size);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with 1D grid indexing (CUDA)");
}
