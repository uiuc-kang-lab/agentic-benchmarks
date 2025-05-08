#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that optimizes workload distribution across the threads while
// ensuring cumulative sum operation's correctness.
// Each thread is responsible for a single column element, and work is
// distributed across multiple blocks to maximize GPU utilization.
__global__ void cumsum_balanced_kernel(const float* __restrict__ input, float* output,
                                       int stride, int total_size) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int idx = thread_id; idx < total_size; idx += total_threads) {
        int outer_inner_idx = idx % (stride * stride);
        int outer_idx = outer_inner_idx / stride;
        int inner_idx = outer_inner_idx % stride;
        int base_idx = (idx - outer_inner_idx) + outer_idx * stride * stride + inner_idx;

        float sum = 0.0f;
        for (int s = 0; s < stride; ++s) {
            int global_idx = base_idx + s * stride;
            sum += __ldg(&input[global_idx]);
            output[global_idx] = sum;
        }
    }
}

// Host function sets up inputs, output tensor, and kernel launch
// parameters before executing the kernel
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
    int total_size = outer_size * inner_size * stride;

    // Configure block and grid sizes for optimal kernel execution
    int threads_per_block = 256;
    int number_of_blocks = (total_size + threads_per_block - 1) / threads_per_block;

    cumsum_balanced_kernel<<<number_of_blocks, threads_per_block>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), stride, total_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with balanced workload");
}