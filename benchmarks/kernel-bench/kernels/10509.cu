#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel with loop unrolling applied to the inner accumulation loop
__global__ void cumsum_kernel_unroll(const float* input, float* output, int stride, int inner_size) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    // Each block processes one outer index, and each thread in the block works on one inner index
    if (inner_idx < inner_size) {
        float sum = 0.0f;
        // Use loop unrolling to reduce overhead of dynamic loop iteration
        #pragma unroll
        for (int i = 0; i < stride; ++i) {
            int idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
            sum += input[idx];
            output[idx] = sum;
        }
    }
}

// Forward function: computes cumulative sum along specified dimension
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

    // Launch the kernel: one block per outer element, inner_size threads per block
    cumsum_kernel_unroll<<<outer_size, inner_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), stride, inner_size
    );
    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with loop unrolling");
}
