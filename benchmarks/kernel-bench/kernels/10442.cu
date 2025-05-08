#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void optimized_cumsum_kernel(const float* __restrict__ input, float* output, int dim, int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    // Use shared memory to reduce global memory access
    extern __shared__ float shared_sum[];

    if (outer_idx < outer_size) {
        float sum = 0.0f;
        for (int i = 0; i < stride; ++i) {
            int idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
            if(inner_idx < inner_size) {
                sum += __ldg(&input[idx]);  // Use __ldg for read-only access
                shared_sum[inner_idx] = sum;
                __syncthreads();
                output[idx] = shared_sum[inner_idx];
            }
        }
    }
}

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

    // Launch with a reasonable number of threads and shared memory
    int threads = min(inner_size, 1024);
    size_t shared_memory_size = threads * sizeof(float);
    optimized_cumsum_kernel<<<outer_size, threads, shared_memory_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), dim, outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA cumulative sum");
}