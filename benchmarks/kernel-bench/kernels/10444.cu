#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel performs a cumulative sum along the specified dimension
// Each block handles one 'outer' index and each thread processes one or more 'inner' indices
// No __syncthreads() is used because threads work completely independently
__global__ void cumsum_kernel_min_sync(const float* __restrict__ input, float* output, 
                                         int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    // Process inner indices with thread striding to cover all elements
    for (int inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
        float sum = 0.0f;
        int base = outer_idx * stride * inner_size + inner_idx;

        // Unroll the loop along the cumulative sum dimension for performance
        #pragma unroll 16
        for (int s = 0; s < stride; s++) {
            int idx = base + s * inner_size;
            sum += __ldg(&input[idx]);  // Efficient read-only cache access
            output[idx] = sum;
        }
    }
}

// Host function: sets up dimensions and launches the kernel
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= x.size(i);
    }

    int stride = x.size(dim);

    // Choose thread count to efficiently cover inner_size, capped at 256
    int threads = (inner_size < 256) ? inner_size : 256;

    // Launch one block per outer index
    cumsum_kernel_min_sync<<<outer_size, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum minimal synchronization");
}
