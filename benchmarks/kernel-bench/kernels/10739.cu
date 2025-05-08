#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Adaptive CUDA kernel for reverse cumulative sum using warp-level primitives and grid strategies
// that adaptively choose between 1D and 2D grid based on input size and dimensions.

__global__ void reverse_cumsum_adaptive_kernel(const float* input, float* output, int row_size, int rows) {
    // Determine if we are using a 1D or 2D grid strategy
    int tid = threadIdx.x;
    int row = (gridDim.y == 1) ? blockIdx.x : (blockIdx.x + blockIdx.y * gridDim.x);
    int gridSize = gridDim.x * gridDim.y;

    if (gridDim.y == 1) { // Single dimension grid approach
        // This block processes a single row
        row = blockIdx.x;
    if (tid >= row_size) return;
        
        int idx = row * row_size + (row_size - 1 - tid);
        float val = input[idx];

        unsigned mask = __activemask();
        
        #pragma unroll
        for (int offset = 1; offset < row_size; offset *= 2) {
            float shfl_val = __shfl_down_sync(mask, val, offset);
            if (tid + offset < row_size) {
                val += shfl_val;
            }
        }
        
        output[idx] = val;
    } else {
        // Grid-stride loop for processing multiple rows
        for (; row < rows; row += gridSize) {
            int index = row * row_size + (row_size - 1 - tid);
            float val = tid < row_size ? input[index] : 0.0f;
            unsigned mask = (1U << row_size) - 1;
            
            #pragma unroll
            for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
                float y = __shfl_down_sync(mask, val, offset);
                val += (tid + offset < row_size) * y;
            }
            if (tid < row_size) {
                output[index] = val;
            }
        }
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    // Ensure the tensor is contiguous and on CUDA
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    x = x.contiguous();

    if (x.scalar_type() != at::kFloat || dim != x.dim() - 1 || x.size(dim) > WARP_SIZE) {
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        return cumsum.flip(dim);
    }

    auto output = at::empty_like(x);
    int row_size = x.size(dim);
    int rows = x.numel() / row_size;

    if (rows > 512) { // Choose 2D grid strategy for larger datasets
        int gridDimX = min(64, rows);
        int gridDimY = (rows + gridDimX - 1) / gridDimX;
        dim3 grid(gridDimX, gridDimY);
        dim3 block(row_size);

        const float* input_ptr = x.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        reverse_cumsum_adaptive_kernel<<<grid, block, 0, stream>>>(input_ptr, output_ptr, row_size, rows);

    } else { // 1D grid for smaller datasets
        dim3 grid(rows);
        dim3 block(row_size);

        const float* input_ptr = x.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        reverse_cumsum_adaptive_kernel<<<grid, block, 0, stream>>>(input_ptr, output_ptr, row_size, rows);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with adaptive grid indexing (CUDA)");
}