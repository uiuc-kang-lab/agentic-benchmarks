#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define BLOCK_SIZE 256

__global__ void cumsum_kernel_coalesced(const float* input, float* output, int outer_size, int inner_size, int stride) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int outer_idx = blockIdx.x;
    const int elements_per_thread = (inner_size + blockDim.x - 1) / blockDim.x;
    const int base_idx = outer_idx * stride * inner_size;
    
    // Process elements in a coalesced manner
    for (int i = 0; i < stride; ++i) {
        // Each thread processes multiple elements with stride equal to block size
        for (int j = 0; j < elements_per_thread; ++j) {
            const int inner_idx = tid + j * blockDim.x;
            if (inner_idx < inner_size) {
                const int current_idx = base_idx + i * inner_size + inner_idx;
                float val = input[current_idx];
                
                // Load value into shared memory
                shared_data[tid] = val;
                __syncthreads();
                
                // Compute running sum for this position
                float sum = 0.0f;
                for (int k = 0; k <= i; ++k) {
                    sum += shared_data[tid];
                }
                
                // Store result
                output[current_idx] = sum;
                __syncthreads();
            }
        }
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
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(outer_size);
    
    cumsum_kernel_coalesced<<<grid, block>>>(
        x.data_ptr<float>(), 
        output.data_ptr<float>(), 
        outer_size, 
        inner_size, 
        stride
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with coalesced memory access");
}