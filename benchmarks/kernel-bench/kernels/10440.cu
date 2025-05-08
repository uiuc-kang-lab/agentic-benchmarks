#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Combined kernel: flatten outer and inner dimensions to one index for increased parallelism
__global__ void combined_cumsum_kernel(const float* __restrict__ input, float* output, int inner_size, int stride, int total_lines) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Use grid-stride loop to cover all independent lines
    while (idx < total_lines) {
        // Decompose line index into outer and inner indices
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;

        // Compute base pointer for the cumulative sum along the stride dimension
        int base = outer_idx * stride * inner_size + inner_idx;
        float sum = 0.f;

        // Compute cumulative sum along the stride dimension
        #pragma unroll
        for (int i = 0; i < stride; ++i) {
            int offset = base + i * inner_size;
            // Use __ldg for read-only access to leverage L1/texture cache
            sum += __ldg(&input[offset]);
            output[offset] = sum;
        }
        idx += blockDim.x * gridDim.x;
    }
}

// Host function launching the kernel
// This function supports cumulative sum along the specified dimension.
// It flattens the outer and inner dimensions to maximize occupancy and parallelism
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    
    // Allocate output tensor
    auto output = torch::empty_like(x);
    
    int ndim = x.dim();
    // Normalize negative dimension
    dim = (dim + ndim) % ndim;

    // Compute outer_size and inner_size following the given dimension
    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= x.size(i);
    }

    // The stride represents the size along the cumulative sum dimension
    int stride = x.size(dim);

    // Total independent lines for cumulative sum (each corresponding to a fixed outer and inner index)
    int total_lines = outer_size * inner_size;

    // Launch configuration using a grid-stride loop over all lines
    int threads = 256;
    int blocks = (total_lines + threads - 1) / threads;
    
    combined_cumsum_kernel<<<blocks, threads>>>(x.data_ptr<float>(), output.data_ptr<float>(), inner_size, stride, total_lines);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient combined CUDA cumulative sum");
}
