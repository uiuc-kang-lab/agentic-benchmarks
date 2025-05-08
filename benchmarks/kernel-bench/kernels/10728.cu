#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Optimized CUDA kernel for reverse cumulative sum using warp-level primitives
// with improved thread and block indexing. This kernel assumes that the cumulative sum
// is performed on the last dimension, that the tensor is contiguous, of type float,
// and that the size along that dimension does not exceed the warp size.
// The kernel maps the rows (all dimensions except the last) onto a 2D grid, with each block
// processing multiple rows through a grid-stride loop to better distribute work and improve occupancy.

__global__ void reverse_cumsum_warp_kernel(const float* input, float* output, int row_size, int rows) {
    // Flatten 2D grid index to a single block id
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int gridSize = gridDim.x * gridDim.y;

    // Each block processes multiple rows via grid-stride loop
    for (int row = blockId; row < rows; row += gridSize) {
        int tid = threadIdx.x;
        // Read element in reverse order - no need to check tid < row_size since row_size <= WARP_SIZE
        int index = row * row_size + (row_size - 1 - tid);
        float val = tid < row_size ? input[index] : 0.0f;
        
        // Create mask only for active threads within row_size
        unsigned mask = (1U << row_size) - 1;
        
        // Perform an inclusive scan within the warp using warp shuffles
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            float y = __shfl_down_sync(mask, val, offset);
            // Remove conditional by using the mask to handle boundary
            val += (tid + offset < row_size) * y;
        }
        
        // Write the computed reverse cumulative sum back in the correct order
        if (tid < row_size) {
            output[index] = val;
        }
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    // Ensure the tensor is contiguous and on CUDA
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    x = x.contiguous();

    // For correctness, we require that the tensor is of type float,
    // the operation is on the last dimension, and its size does not exceed WARP_SIZE
    if (x.scalar_type() != at::kFloat || dim != x.dim() - 1 || x.size(dim) > WARP_SIZE) {
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        return cumsum.flip(dim);
    }

    // Prepare output tensor
    auto output = at::empty_like(x);
    int row_size = x.size(dim);
    int rows = x.numel() / row_size;

    // Configure 2D grid for improved thread and block mapping
    // Choose a modest number for gridDim.x to ensure good occupancy
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
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with optimized indexing (CUDA)");
}
