#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define BLOCK_SIZE 256

__global__ void cumsum_kernel_shared(const float* __restrict__ input, float* __restrict__ output,
                                   int outer_size, int inner_size, int stride) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    float running_sum = 0.0f;
    
    for (int i = 0; i < stride; ++i) {
        int idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
        
        // Load current value into shared memory
        shared_data[inner_idx] = __ldg(&input[idx]);
        __syncthreads();
        
        // Update running sum
        running_sum += shared_data[inner_idx];
        output[idx] = running_sum;
        
        // Only synchronize if we need to ensure shared memory is consistent for next iteration
        if (i < stride - 1) {
            __syncthreads();
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
    
    dim3 block(min(inner_size, BLOCK_SIZE));
    dim3 grid(outer_size);
    
    cumsum_kernel_shared<<<grid, block>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with shared memory");
}