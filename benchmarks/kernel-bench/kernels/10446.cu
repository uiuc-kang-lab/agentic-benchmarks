#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel minimizes warp divergence by precomputing the exact number of iterations
// each thread must execute over the inner dimension. By launching with blockDim.x set to
// min(inner_size, 1024), we ensure that all threads in a warp follow a uniform control flow,
// avoiding per-iteration conditional checks for validity.
__global__ void cumsum_nodiv_kernel(const float* __restrict__ input, float* output, int outer_size, int inner_size, int stride) {
    // Each block handles one 'outer' index
    int outer_idx = blockIdx.x;
    int threadId = threadIdx.x;

    // When inner_size <= 1024, blockDim.x is set to inner_size so all threads are valid.
    // When inner_size > 1024, we use a grid-stride loop. The number of iterations for each thread
    // is computed as: n_iters = floor((inner_size - threadId - 1)/blockDim.x) + 1.
    int n_iters = ((inner_size - threadId - 1) / blockDim.x) + 1;

    // Loop over assigned inner indices without per-iteration validity checks
    for (int i = 0; i < n_iters; i++) {
        int inner_idx = threadId + i * blockDim.x;  // Guaranteed to be < inner_size by n_iters calcuation
        int base_idx = outer_idx * stride * inner_size + inner_idx;
        float sum = 0.0f;

        #pragma unroll 16
        for (int s = 0; s < stride; s++) {
            int idx = base_idx + s * inner_size;
            sum += __ldg(&input[idx]);  // Use read-only cache intrinsic
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
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= x.size(i);
    }

    int stride = x.size(dim);

    // Launch configuration: one block per outer index. For inner dimension,
    // we choose the number of threads as min(inner_size, 1024) to avoid divergent branches
    // within warps.
    int threads = (inner_size < 1024) ? inner_size : 1024;
    cumsum_nodiv_kernel<<<outer_size, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with minimized warp divergence");
}
