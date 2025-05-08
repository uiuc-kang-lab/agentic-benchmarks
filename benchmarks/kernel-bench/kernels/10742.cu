#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void reverse_cumsum_kernel_optimized(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              const int row_size,
                                              const int total_elements) {
    // Calculate global thread index
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    // Calculate row and column indices
    const int row = gid / row_size;
    const int col = gid % row_size;
    
    if (gid < total_elements) {
        // Calculate the reversed column index within the row
        const int rev_col = row_size - 1 - col;
        const int idx = row * row_size + rev_col;
        
        // Load input value
        float val = input[idx];
        
        // Calculate warp lane mask for active threads in this row
        const unsigned mask = __ballot_sync(0xffffffff, true);
        
        // Perform warp-level scan
        #pragma unroll
        for (int offset = 1; offset < row_size; offset *= 2) {
            const float n = __shfl_down_sync(mask, val, offset);
            if ((col + offset) < row_size) {
                val += n;
            }
        }
        
        // Store result
        output[idx] = val;
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    x = x.contiguous();

    if (x.scalar_type() != at::kFloat || dim != x.dim() - 1 || x.size(dim) > WARP_SIZE) {
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        return cumsum.flip(dim);
    }

    auto output = at::empty_like(x);
    const int row_size = x.size(dim);
    const int total_elements = x.numel();
    
    // Calculate grid dimensions
    const int num_threads = BLOCK_SIZE;
    const int num_blocks = (total_elements + num_threads - 1) / num_threads;
    
    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    reverse_cumsum_kernel_optimized<<<num_blocks, num_threads, 0, stream>>>(
        input_ptr, output_ptr, row_size, total_elements
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with optimized thread mapping (CUDA)");
}