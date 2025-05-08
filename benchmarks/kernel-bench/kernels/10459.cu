#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that uses grid-stride and thread-stride loops to handle workloads larger than available threads
__global__ void cumsum_stride_loop_kernel(const float* __restrict__ input, float* output,
                                            int outer_size, int inner_size, int stride) {
    // Iterate over the outer dimension with a grid-stride loop
    for (int outer = blockIdx.x; outer < outer_size; outer += gridDim.x) {
        // Iterate over the inner dimension with a thread-stride loop
        for (int inner = threadIdx.x; inner < inner_size; inner += blockDim.x) {
            float sum = 0.0f;
            // Loop sequentially over the cumulative sum (stride) dimension
            for (int s = 0; s < stride; s++) {
                int idx = outer * (stride * inner_size) + s * inner_size + inner;
                sum += __ldg(&input[idx]);
                output[idx] = sum;
            }
        }
    }
}

// Host function to set up tensor dimensions and launch the kernel
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

    // Determine grid and block dimensions; use a grid-stride loop in the kernel
    int blocks = (outer_size < 1024) ? outer_size : 1024;
    int threads = 256;

    cumsum_stride_loop_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        outer_size,
        inner_size,
        stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with stride loops for large workloads");
}
