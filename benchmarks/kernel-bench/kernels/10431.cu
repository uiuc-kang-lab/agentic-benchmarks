#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void cumsum_kernel_uniform(const float* input, float* output, int dim, int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < stride; ++i) {
        int idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
        if (inner_idx < inner_size && outer_idx < outer_size) {
            sum += input[idx];
            output[idx] = sum;
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

    cumsum_kernel_uniform<<<outer_size, inner_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), dim, outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum");
}