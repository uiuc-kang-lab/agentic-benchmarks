#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// Optimized kernel combining thread striding for large inner_size and __ldg for efficient read-only access
__global__ void cumsum_kernel_optimized(const float* __restrict__ input, float* output, int outer_size, int inner_size, int stride) {
    // Each block handles one 'outer' dimension index
    int outer_idx = blockIdx.x;

    // Loop over the inner index using thread striding to cover large inner_size
    for (int inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
        float sum = 0.0f;
        // Base index corresponding to the beginning of the cumulative sum for this (outer, inner) pair
        int base_idx = outer_idx * stride * inner_size + inner_idx;

        // Unroll the loop along the cumulative sum dimension for performance
        #pragma unroll 16
        for (int s = 0; s < stride; ++s) {
            int idx = base_idx + s * inner_size;
            // Use __ldg to optimize read-only memory access
            sum += __ldg(&input[idx]);
            output[idx] = sum;
        }
    }
}

// Host function forwarding the tensor, computing sizes and launching the optimized kernel

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    
    auto output = torch::empty_like(x);
    int ndim = x.dim();
    // Ensure dimension is within range
    dim = (dim + ndim) % ndim;

    // Compute the outer_size as product of dimensions before the specified dimension
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= x.size(i);
    }

    // Compute the inner_size as product of dimensions after the specified dimension
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= x.size(i);
    }

    // The stride (number of elements in the cumulative sum dimension)
    int stride = x.size(dim);

    // Choose number of threads for the inner dimension, capped by 1024
    int threads = (inner_size < 1024) ? inner_size : 1024;

    // Launch one block per outer index
    cumsum_kernel_optimized<<<outer_size, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA cumulative sum");
}
