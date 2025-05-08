#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void cumsum_kernel_optimized(const float* input, float* output, int dim, int outer_size, int inner_size, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int idx = tid; idx < outer_size * inner_size; idx += total_threads) {
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;
        float sum = 0.0f;
        for (int i = 0; i < stride; ++i) {
            int linear_idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
            sum += input[linear_idx];
            output[linear_idx] = sum;
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
    int threads = 256;
    int blocks = (outer_size * inner_size + threads - 1) / threads;

    cumsum_kernel_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), dim, outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA cumulative sum");
}