#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel using stride loops to handle inner_size > blockDim.x
__global__ void cumsum_kernel(const float* input, float* output, int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;
    
    // Loop over inner indices using stride looping
    for (int inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
        float sum = 0.0f;
        // Compute cumulative sum along the stride dimension
        for (int s = 0; s < stride; s++) {
            int idx = outer_idx * stride * inner_size + s * inner_size + inner_idx;
            sum += input[idx];
            output[idx] = sum;
        }
    }
}

// Host function interfacing with PyTorch
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    
    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;
    
    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= x.size(i);
    }
    
    int stride = x.size(dim);
    
    // Use a grid with outer_size blocks and a sufficient number of threads to cover inner_size
    int threads = (inner_size < 1024) ? inner_size : 1024;
    cumsum_kernel<<<outer_size, threads>>>(x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with stride loops");
}
