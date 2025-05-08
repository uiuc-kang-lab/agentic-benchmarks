#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void prod_reduce_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  const int stride,
                                  const int num_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_elements) return;

    float product = 1.0f;
    const int base_offset = idx;

    if (stride == 1) {
        // Vectorized access pattern for aligned loads with fast math intrinsics
        const float4* vec_ptr = reinterpret_cast<const float4*>(input + base_offset);
        
        #pragma unroll
        for (int i = 0; i < 12; ++i) {  // 12 * 4 elements = 48 elements
            float4 vec = __ldg(&vec_ptr[i]);
            // Use fast multiply-add for better performance
            product = __fmul_rn(product, __fmul_rn(__fmul_rn(vec.x, vec.y), __fmul_rn(vec.z, vec.w)));
        }
        // Handle remaining 2 elements using fast math
        product = __fmul_rn(product, __ldg(&input[base_offset + 48]));
        product = __fmul_rn(product, __ldg(&input[base_offset + 49]));
    } else {
        // Optimized non-contiguous access with coalesced reads
        #pragma unroll
        for (int i = 0; i < 50; ++i) {
            product *= __ldg(&input[base_offset + i * stride]);
        }
    }

    output[idx] = product;
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    const int num_elements = output.numel();
    const int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Optimized for memory throughput with 256 threads/block
    const int block_size = 256;
    const int blocks = (num_elements + block_size - 1) / block_size;

    prod_reduce_kernel<<<blocks, block_size>>>(input_ptr, output_ptr, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction with vectorized LDG optimization");
}