#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void grid_stride_reduction_kernel(const float* input, float* output, int stride, int num_elements) {
    constexpr int UNROLL_FACTOR = 50;
    int grid_stride = blockDim.x * gridDim.x;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < num_elements; 
         idx += grid_stride) {
        float product = 1.0f;
        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; ++i) {
            product = __fmul_rn(product, input[idx + i * stride]);
        }
        output[idx] = product;
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;
    blocks = min(blocks, 256);  // Better occupancy with multiple smaller blocks

    grid_stride_reduction_kernel<<<blocks, threads>>>(input_ptr, output_ptr, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Grid-strided product reduction (CUDA)");
}