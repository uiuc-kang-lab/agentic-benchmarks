#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel loads global memory in 128-bit aligned chunks using float4 and __ldg()
// to optimize read-only accesses. It processes most of the data in groups of 4 floats
// and then handles any remaining elements individually.
__global__ void softsign_kernel_aligned(const float* __restrict__ x, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    // Process groups of 4 floats (128 bits) at a time
    int n4 = num_elements / 4; // number of float4 groups
    const float4* in4 = reinterpret_cast<const float4*>(x);
    float4* out4 = reinterpret_cast<float4*>(out);

    for (int i = idx; i < n4; i += gridSize) {
        // Use __ldg for read-only data
        float4 data = __ldg(&in4[i]);
        data.x = data.x / (1.0f + fabsf(data.x));
        data.y = data.y / (1.0f + fabsf(data.y));
        data.z = data.z / (1.0f + fabsf(data.z));
        data.w = data.w / (1.0f + fabsf(data.w));
        out4[i] = data;
    }

    // Process any remaining elements if num_elements is not a multiple of 4
    int rem = n4 * 4;
    for (int i = rem + idx; i < num_elements; i += gridSize) {
        float val = __ldg(&x[i]);
        out[i] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    // Choosing 1024 threads per block; grid size determined by total number of elements
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    softsign_kernel_aligned<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with aligned loads using __ldg (CUDA)");
}
