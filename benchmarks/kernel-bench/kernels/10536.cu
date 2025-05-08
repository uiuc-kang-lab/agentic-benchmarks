#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized cumulative sum kernel using shared memory to reduce global memory accesses
__global__ void cumsum_kernel_shared(const float* __restrict__ input, float* __restrict__ output,
                                     int outer_size, int inner_size, int stride) {
    extern __shared__ float shared_mem[];
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        float sum = 0.0f;
        for (int i = 0; i < stride; ++i) {
            int idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
            shared_mem[inner_idx] = input[idx];
            if (inner_idx == 0) { __syncthreads(); }

            sum += shared_mem[inner_idx];
            output[idx] = sum;
            if (inner_idx == 0) { __syncthreads(); }
        }
    }
}

// Forward function: computes cumulative sum along the specified dimension
// using shared memory to reduce global memory accesses
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
    cumsum_kernel_shared<<<outer_size, inner_size, inner_size * sizeof(float)>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA cumulative sum using shared memory");
}