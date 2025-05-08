#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void prod_reduce_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                 int stride, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        float product = 1.0f;
        
        if (stride == 1) {  // Contiguous case - can use vectorized loads
            const float4* input_vec = reinterpret_cast<const float4*>(input + idx);
            #pragma unroll
            for (int i = 0; i < 12; ++i) {  // Process 48 elements (12 float4s)
                float4 vec = __ldg(&input_vec[i]);
                product *= vec.x * vec.y * vec.z * vec.w;
            }
            // Handle remaining 2 elements
            product *= __ldg(&input[idx + 48]);
            product *= __ldg(&input[idx + 49]);
        } else {  // Non-contiguous case
            #pragma unroll
            for (int i = 0; i < 50; ++i) {
                product *= __ldg(&input[idx + i * stride]);
            }
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

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    prod_reduce_kernel<<<blocks, threads>>>(input_ptr, output_ptr, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}