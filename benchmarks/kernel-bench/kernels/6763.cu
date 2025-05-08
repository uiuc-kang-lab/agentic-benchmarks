#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized kernel using __ldg for read-only accesses and 128-bit aligned vectorized loads
__global__ void prod_reduce_kernel_optimized(const float* __restrict__ input, float* __restrict__ output, 
                                               int dim_size, int stride, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float product = 1.0f; __shared__ float shared_product[1024];
        // Use vectorized loads if conditions are met:
        // 1. The reduction dimension is contiguous (stride == 1).
        // 2. The reduction size is a multiple of 4 so we can load 4 floats (128 bits) at a time.
        // 3. The input base pointer is 16-byte aligned and the thread's starting offset is also aligned.
        if (stride == 1 && (dim_size % 4 == 0) && (((uintptr_t)input & 0xF) == 0) && ((idx & 3) == 0)) {
            const float4* input_vec = reinterpret_cast<const float4*>(input + idx);
            int num_vec = dim_size / 4;
            for (int i = 0; i < num_vec; i++) {
                // Using __ldg() to load a float4 from global read-only memory
                float4 vec = __ldg(&input_vec[i]);
                product *= vec.x * vec.y * vec.z * vec.w;
            }
        } else {
            // Fallback to scalar loads using __ldg() for each element
            for (int i = 0; i < dim_size; ++i) {
                product *= __ldg(input + idx + i * stride);
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

    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    prod_reduce_kernel_optimized<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized product reduction over a dimension (CUDA)");
}
