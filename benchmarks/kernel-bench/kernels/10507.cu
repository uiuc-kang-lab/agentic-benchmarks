#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ float compute_cumsum(const float* input, int idx, int stride, int inner_size) {
    float sum = 0.0f;
    for (int i = 0; i < stride; ++i) {
        int current_idx = idx + i * inner_size;
        sum += input[current_idx];
    }
    return sum;
}

__global__ void cumsum_kernel(const float* input, float* output, int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        int idx = outer_idx * stride * inner_size + inner_idx;
        float sum = compute_cumsum(input, idx, stride, inner_size);
        for (int i = 0; i < stride; ++i) {
            int current_idx = idx + i * inner_size;
            output[current_idx] = sum;
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

    cumsum_kernel<<<outer_size, inner_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum");
}