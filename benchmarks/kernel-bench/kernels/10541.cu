#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void cumsum_warp_optimized(const float* __restrict__ input, float* __restrict__ output,
                                      int outer_size, int inner_size, int stride) {
    const int total_elements = outer_size * inner_size;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < total_elements) {
        const int outer_idx = tid / inner_size;
        const int inner_idx = tid % inner_size;
        
        float sum = 0.0f;
        for (int i = 0; i < stride; ++i) {
            const int idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
            sum += __ldg(&input[idx]);
            output[idx] = sum;
        }
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto output = torch::empty_like(x);
    const int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= x.size(i);
    }

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= x.size(i);
    }

    const int stride = x.size(dim);
    const int total_elements = outer_size * inner_size;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    cumsum_warp_optimized<<<grid_size, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(),
                                                   outer_size, inner_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized CUDA cumulative sum");
}
