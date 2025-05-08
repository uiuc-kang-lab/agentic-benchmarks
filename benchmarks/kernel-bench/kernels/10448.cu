#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel computes the cumulative sum along a specified dimension
// It uses a stride loop over the inner dimension to handle cases where inner_size
// is larger than the number of available threads. This ensures that all elements are processed
// correctly even for large workloads. The loop over the cumulative dimension (stride) is unrolled
// to reduce loop overhead, and __ldg is used for efficient read-only memory access.
__global__ void cumsum_stride_kernel(const float* __restrict__ input, float* output, int outer_size, int inner_size, int stride) {
    // Each block handles one 'outer' index
    int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    // Base offset for the current outer index
    int base_offset = outer_idx * stride * inner_size;

    // Use a stride loop to cover all inner indices
    for (int inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
        float sum = 0.0f;
        // Unroll the loop over the cumulative (stride) dimension for performance
        #pragma unroll 16
        for (int s = 0; s < stride; ++s) {
            int idx = base_offset + s * inner_size + inner_idx;
            sum += __ldg(&input[idx]);
            output[idx] = sum;
        }
    }
}

// Host function interfacing with PyTorch
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

    // Choose the number of threads per block as the minimum of inner_size and a reasonable cap
    int threads = (inner_size < 256) ? inner_size : 256;

    // Launch one block per outer index
    cumsum_stride_kernel<<<outer_size, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with stride loops for boundary handling");
}
