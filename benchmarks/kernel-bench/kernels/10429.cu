#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define BLOCK_SIZE 256

__global__ void cumsum_kernel(const float* input, float* output, int dim, int outer_size, int inner_size, int stride) {
    __shared__ float shared_sum[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int outer_idx = blockIdx.x;
    int inner_batch = (inner_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int batch = 0; batch < inner_batch; batch++) {
        int inner_idx = batch * BLOCK_SIZE + tid;
        if (inner_idx < inner_size && outer_idx < outer_size) {
            float sum = 0.0f;
            for (int i = 0; i < stride; ++i) {
                int idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
                sum += input[idx];
                output[idx] = sum;
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
    
    cumsum_kernel<<<grid, block>>>(x.data_ptr<float>(), output.data_ptr<float>(), dim, outer_size, inner_size, stride);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum");
}