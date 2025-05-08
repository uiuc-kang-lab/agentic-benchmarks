#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// CUDA kernel using warp shuffles for reverse cumulative sum
// This kernel assumes that the cumulative sum is performed on the last dimension,
// that the tensor is contiguous, of type float, and that the size along that dimension
// does not exceed the warp size (WARP_SIZE).
__global__ void reverse_cumsum_warp_kernel(const float* input, float* output, int row_size, int rows) {
    // Flatten 2D grid index to a single block id
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int gridSize = gridDim.x * gridDim.y;

    // Process multiple rows via grid-stride loop
    for (int row = blockId; row < rows; row += gridSize) {
        int tid = threadIdx.x;
        // Compute index for reverse order: element from end to beginning
        int index = row * row_size + (row_size - 1 - tid);
        float val = (tid < row_size) ? input[index] : 0.0f;

        // Mask for active threads within the warp; row_size is assumed to be <= WARP_SIZE
        unsigned mask = (1U << row_size) - 1;

        // Inclusive scan within a warp using shuffle down reductions
        #pragma unroll
        for (int offset = 1; offset < row_size; offset *= 2) {
            float y = __shfl_down_sync(mask, val, offset);
            if (tid + offset < row_size) {
                val += y;
            }
        }

        // Write the computed reverse cumulative sum back
        if (tid < row_size) {
            output[index] = val;
        }
    }
}

// Combined reverse cumulative sum function
// This function checks if the input tensor meets optimized conditions:
//   - Tensor is of type float
//   - The operation is applied on the last dimension
//   - The size along that dimension does not exceed WARP_SIZE
// If the conditions are met, the custom CUDA kernel is launched.
// Otherwise, the function falls back to a method using tensor flip and built-in cumsum for correctness.

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    x = x.contiguous();

    // Check if conditions for optimized kernel are met
    if (x.scalar_type() != at::kFloat || dim != x.dim() - 1 || x.size(dim) > WARP_SIZE) {
        // Fallback: use flip, cumsum, and flip back
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        return cumsum.flip(dim);
    }

    // Prepare output tensor
    auto output = at::empty_like(x);
    int row_size = x.size(dim);
    int rows = x.numel() / row_size;

    // Configure a 2D grid for improved thread and block mapping
    int gridDimX = (rows < 64) ? rows : 64;
    int gridDimY = (rows + gridDimX - 1) / gridDimX;
    dim3 grid(gridDimX, gridDimY);
    dim3 block(row_size);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    reverse_cumsum_warp_kernel<<<grid, block, 0, stream>>>(input_ptr, output_ptr, row_size, rows);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Optimized reverse cumulative sum (CUDA)");
}
