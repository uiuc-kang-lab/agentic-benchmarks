#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Modular device function to compute the linear index for accessing the tensor
__device__ inline int get_index(int outer_idx, int i, int inner_idx, int inner_size, int stride) {
    return outer_idx * (stride * inner_size) + i * inner_size + inner_idx;
}

// Modular device function to process the cumulative sum for one (outer, inner) position
__device__ void process_cumsum(const float* input, float* output, int outer_idx, int inner_idx, int inner_size, int stride) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < stride; i++) {
        int index = get_index(outer_idx, i, inner_idx, inner_size, stride);
        sum += input[index];
        output[index] = sum;
    }
}

// CUDA kernel using a grid-stride loop for inner indices
__global__ void cumsum_kernel(const float* input, float* output, int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    // Process each inner index using a grid-stride loop
    for (int inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
         process_cumsum(input, output, outer_idx, inner_idx, inner_size, stride);
    }
}

// Host function interfacing with PyTorch and launching the kernel
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
    int threads = 256;

    dim3 grid(outer_size);
    dim3 block(threads);

    cumsum_kernel<<<grid, block>>>(x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular CUDA cumulative sum");
}
