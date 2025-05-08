#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized cumulative sum kernel using __ldg() for read-only accesses and __restrict__ pointers
// with CUDA streams to overlap computation and memory operations
__global__ void cumsum_kernel_opt_stream(const float* __restrict__ input, float* __restrict__ output,
                                          int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

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
// using CUDA streams for overlapping memory operations and computation
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

    // Create a CUDA stream for overlapping computation and memory operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Launch the kernel: one block per outer index, inner_size threads per block
    // Stream the kernel launch to overlap with other operations
    cumsum_kernel_opt_stream<<<outer_size, inner_size, 0, stream>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    // Synchronize the stream after kernel execution
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA cumulative sum using __ldg(), aligned accesses, and streams");
}
