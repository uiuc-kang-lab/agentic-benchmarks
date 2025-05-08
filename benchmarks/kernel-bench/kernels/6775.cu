#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Use constant memory to store precomputed offsets (i * stride) for reduction dimension
__constant__ int d_offsets[50];

__global__ void prod_reduce_kernel(const float* input, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float product = 1.0f;
        #pragma unroll
        for (int i = 0; i < 50; ++i) {
            product *= input[idx + d_offsets[i]];
        }
        output[idx] = product;
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    TORCH_CHECK(dim_size == 50, "Dimension size must be 50 for this kernel");
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    // Precompute offsets for the reduction dimension and copy to constant memory
    int offsets_host[50];
    for (int i = 0; i < 50; ++i) {
        offsets_host[i] = i * stride;
    }
    cudaMemcpyToSymbol(d_offsets, offsets_host, 50 * sizeof(int), 0, cudaMemcpyHostToDevice);

    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    prod_reduce_kernel<<<blocks, threads>>>(input_ptr, output_ptr, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Constant Memory Unrolled 50-Product Reduction over a dimension (CUDA)");
}
