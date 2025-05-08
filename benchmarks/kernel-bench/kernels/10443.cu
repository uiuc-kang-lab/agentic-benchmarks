#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel using warp-level primitives for efficient reduction
__global__ void cumsum_kernel_warp(const float* __restrict__ input, float* output, int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        float sum = 0.0f;
        int base_idx = outer_idx * stride * inner_size + inner_idx;

        // Using warp-level primitives for reduction
        #pragma unroll 16
        for (int i = 0; i < stride; ++i) {
            int idx = base_idx + i * inner_size;
            sum += __ldg(&input[idx]); // Use __ldg for read-only access
            sum = __shfl_down_sync(0xFFFFFFFF, sum, 1);
            if (inner_idx % 32 == 0) { // Only write if thread is the first in the warp
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

    cumsum_kernel_warp<<<outer_size, inner_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with warp optimization");
}
