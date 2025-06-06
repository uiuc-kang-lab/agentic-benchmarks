#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void cumsum_kernel_coalesced(const float* __restrict__ input, 
                                      float* __restrict__ output,
                                      int outer_size, int inner_size, int stride) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_idx = bid * blockDim.x + tid;
    
    // Calculate base indices for coalesced access
    const int outer_idx = global_idx / inner_size;
    const int inner_idx = global_idx % inner_size;
    
    if (outer_idx >= outer_size) return;
    
    // Process each slice with coalesced access pattern
    float running_sum = 0.0f;
    const int base_idx = outer_idx * stride * inner_size + inner_idx;
    
    #pragma unroll 4
    for (int i = 0; i < stride; i++) {
        const int curr_idx = base_idx + i * inner_size;
        // Load data in a coalesced manner
        const float val = __ldg(&input[curr_idx]);
        
        // Compute local sum
        running_sum += val;
        
        // Store result with coalesced write
        output[curr_idx] = running_sum;
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    
    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;
    
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= x.size(i);
    }
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= x.size(i);
    }
    
    int stride = x.size(dim);
    
    // Calculate grid and block dimensions for optimal occupancy
    const int total_threads = outer_size * inner_size;
    const int num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cumsum_kernel_coalesced<<<num_blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(), 
        output.data_ptr<float>(),
        outer_size, inner_size, stride
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced memory access CUDA cumulative sum");
}