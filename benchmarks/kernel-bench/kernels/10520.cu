#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized cumulative sum kernel using __ldg() for read-only accesses and __restrict__ pointers
// Assumes that the tensor memory is 128-bit aligned (which is generally true for PyTorch tensors).
__global__ void cumsum_kernel_opt(const float* __restrict__ input, float* __restrict__ output,
                                    int outer_size, int inner_size, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;
    int outer_idx = tid / inner_size;
    int inner_idx = tid % inner_size;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < stride; ++i) {
            int idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
            // Use __ldg() to optimize read-only global memory load
            float val = __ldg(&input[idx]);
            sum += val;
            output[idx] = sum;
        }
    }
}

// Forward function: computes cumulative sum along the specified dimension
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

    // Launch the kernel: one block per outer index, inner_size threads per block
    cumsum_kernel_opt<<<outer_size, inner_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA cumulative sum using __ldg() and aligned accesses");
}
