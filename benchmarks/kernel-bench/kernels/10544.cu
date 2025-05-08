#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized cumulative sum kernel with manual loop unrolling to reduce loop overhead
__global__ void cumsum_kernel_unroll(const float* __restrict__ input, float* __restrict__ output,
                                       int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    int base = outer_idx * stride * inner_size;

    // Each thread processes multiple inner indices in steps of blockDim.x for better occupancy
    for (int inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
        float sum = 0.0f;
        const int unroll_factor = 4;
        int limit = stride - (stride % unroll_factor);

        #pragma unroll
        for (int i = 0; i < limit; i += unroll_factor) {
            int idx0 = base + (i + 0) * inner_size + inner_idx;
            sum += __ldg(&input[idx0]);
            output[idx0] = sum;

            int idx1 = base + (i + 1) * inner_size + inner_idx;
            sum += __ldg(&input[idx1]);
            output[idx1] = sum;

            int idx2 = base + (i + 2) * inner_size + inner_idx;
            sum += __ldg(&input[idx2]);
            output[idx2] = sum;

            int idx3 = base + (i + 3) * inner_size + inner_idx;
            sum += __ldg(&input[idx3]);
            output[idx3] = sum;
        }

        // Process remaining iterations if stride is not divisible by unroll_factor
        for (int i = limit; i < stride; i++) {
            int idx = base + i * inner_size + inner_idx;
            sum += __ldg(&input[idx]);
            output[idx] = sum;
        }
    }
}

// Forward function: computes cumulative sum along the specified dimension
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

    // Launch the kernel: each block processes one outer index, each thread corresponds to an inner index
    cumsum_kernel_unroll<<<outer_size, inner_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        outer_size,
        inner_size,
        stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA cumulative sum with manual loop unrolling");
}
