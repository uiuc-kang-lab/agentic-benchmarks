#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Experiment with block sizes based on hardware. Here we choose 256 as an example.
// We use a 2D grid: grid.x corresponds to outer indices, grid.y covers the inner dimension splits.

const int BLOCK_SIZE = 256;

__global__ void cumsum_kernel_opt(const float* input, float* output, int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        float sum = 0.0f;
        for (int i = 0; i < stride; ++i) {
            int idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
            sum += input[idx];
            output[idx] = sum;
        }
    }
}

// The forward function adapts the grid dimensions based on tensor shape and chosen block size.

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

    // Use a 2D grid: one dimension for outer indices, one for covering the inner dimension based on BLOCK_SIZE.
    dim3 grid(outer_size, (inner_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cumsum_kernel_opt<<<grid, BLOCK_SIZE>>>(x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum optimized with block size experimentation");
}
