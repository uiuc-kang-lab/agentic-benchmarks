#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized cumulative sum kernel focusing on memory coalescing
__global__ void cumsum_kernel_coalesced(const float* __restrict__ input, float* __restrict__ output,
                                         int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    int base_idx = outer_idx * stride * inner_size;

    // Ensure each thread works on consecutive memory elements to maximize coalescing
    for (int i = 0; i < stride; ++i) {
        int idx = base_idx + i * inner_size + threadIdx.x;
        if (threadIdx.x < inner_size) {
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j <= i; ++j) {
                int read_idx = base_idx + j * inner_size + threadIdx.x;
                sum += input[read_idx];
            }
            output[idx] = sum;
        }
    }
}

// Forward function: computes cumulative sum along the specified dimension
// with a focus on coalesced memory accesses for improved performance
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

    // Configure kernel to maximize thread utilization and coalescing
    cumsum_kernel_coalesced<<<outer_size, inner_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with coalesced memory accesses");
}