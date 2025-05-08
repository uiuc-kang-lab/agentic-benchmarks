#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void cumsum_kernel_stride(const float* __restrict__ input, float* output, int dim, int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx < outer_size) {
        int base_idx = outer_idx * stride * inner_size + inner_idx;

        // Use a stride loop to handle larger workloads
        for (int offset = inner_idx; offset < inner_size; offset += blockDim.x) {
            float sum = 0.0f;
            int current_base = outer_idx * stride * inner_size + offset;

            #pragma unroll 16
            for (int i = 0; i < stride; ++i) {
                int idx = current_base + i * inner_size;
                sum += __ldg(&input[idx]);  // Use read-only access for efficiency
                output[idx] = sum;
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

    // Cap the number of threads to 1024 while ensuring full coverage of inner_size
    int threads = (inner_size < 1024) ? inner_size : 1024;

    cumsum_kernel_stride<<<outer_size, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), dim, outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with stride loop");
}
