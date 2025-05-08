#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Compute linear index with coalesced memory access pattern
__device__ __forceinline__ int get_index(int outer_idx, int i, int inner_idx, int inner_size, int stride) {
    return outer_idx * (stride * inner_size) + i * inner_size + inner_idx;
}

// Process cumulative sum using shared memory for frequently accessed data
__global__ void cumsum_kernel(const float* __restrict__ input, float* output, 
                            int outer_size, int inner_size, int stride) {
    extern __shared__ float shared_sum[];
    
    int outer_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (outer_idx >= outer_size) return;

    // Process multiple inner indices per thread using grid-stride loop
    for (int inner_idx = tid; inner_idx < inner_size; inner_idx += blockDim.x) {
        float sum = 0.0f;
        shared_sum[tid] = 0.0f;
        
        #pragma unroll 16
        for (int i = 0; i < stride; ++i) {
            int idx = get_index(outer_idx, i, inner_idx, inner_size, stride);
            sum += __ldg(&input[idx]);  // Use __ldg for read-only cache
            output[idx] = sum;
            
            // Store partial sums in shared memory for potential reuse
            shared_sum[tid] = sum;
        }
        __syncthreads();
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
    
    // Optimize thread block size based on GPU architecture
    const int threads_per_block = 256;
    dim3 grid(outer_size);
    dim3 block(threads_per_block);
    
    // Allocate shared memory for partial sums
    int shared_mem_size = threads_per_block * sizeof(float);
    
    cumsum_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), 
        outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized modular CUDA cumulative sum");
}