#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define warp size
#define WARP_SIZE 32

// Optimized CUDA kernel for reverse cumulative sum using warp-level primitives.
// This kernel assumes that the cumulative sum is performed on the last dimension,
// that the tensor is contiguous, of type float, and that the size along the dimension
// is small (<= WARP_SIZE). For other cases, the host function falls back to the reference
// implementation to ensure correctness.

__global__ void reverse_cumsum_warp_kernel(const float* input, float* output, int row_size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= row_size) return;  // Safety check

    // Compute the index for the element when reading in reverse order
    int index = row * row_size + (row_size - 1 - tid);
    float val = input[index];

    // Perform an inclusive scan within the warp using warp shuffles
    unsigned mask = __activemask();
    for (int offset = 1; offset < row_size; offset *= 2) {
        float y = __shfl_down_sync(mask, val, offset);
        if (tid + offset < row_size) {
            val += y;
        }
    }

    // Write the computed reverse cumulative sum back in the correct order
    output[index] = val;
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    // Ensure the tensor is contiguous and on CUDA
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    x = x.contiguous();

    // For correctness and to use our optimized kernel safely, we require that:
    // 1. The tensor is of type float
    // 2. The cumulative sum is performed along the last dimension
    // 3. The size along that dimension does not exceed the warp size
    if (x.scalar_type() != at::kFloat || dim != x.dim() - 1 || x.size(dim) > WARP_SIZE) {
        // Fallback to the reference implementation using tensor flip and cumsum operations
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        return cumsum.flip(dim);
    }

    // Prepare output tensor
    auto output = at::empty_like(x);

    // The reverse cumulative sum is computed along the last dimension.
    // Number of elements along that dimension
    int row_size = x.size(dim);
    // The remaining dimensions form the number of rows
    int rows = x.numel() / row_size;

    // Launch one block per row and use row_size threads per block
    dim3 grid((rows + WARP_SIZE - 1) / WARP_SIZE);
    dim3 block(row_size);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    reverse_cumsum_warp_kernel<<<grid, block, 0, stream>>>(input_ptr, output_ptr, row_size);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with warp-level primitives (CUDA)");
}
