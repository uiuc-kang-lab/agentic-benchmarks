#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void cumsum_kernel(const float* input, float* output, int total_pairs, int inner_size, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_pairs) return;
    
    int outer_idx = tid / inner_size;
    int inner_idx = tid % inner_size;
    
    float sum = 0.0f;
    for (int i = 0; i < stride; ++i) {
        int idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
        sum += input[idx];
        output[idx] = sum;
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
    int total_pairs = outer_size * inner_size;
    int threads_per_block = 256;
    int blocks = (total_pairs + threads_per_block - 1) / threads_per_block;

    cumsum_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), total_pairs, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum optimized");
}