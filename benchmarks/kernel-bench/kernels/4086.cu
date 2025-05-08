#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Declare a constant memory variable to store the alpha value
__constant__ float elu_alpha;

__global__ void elu_kernel_constant(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;  // total threads in grid
    int total_stride = stride * 4;         // each thread processes groups of 4 elements

    for (int tid_base = idx * 4; tid_base < n; tid_base += total_stride) {
        int remaining = n - tid_base;
        if (remaining >= 4) {
            // Vectorized load
            const float4 vec = *reinterpret_cast<const float4*>(x + tid_base);
            
            // Compute ELU
            float4 result;
            result.x = (vec.x > 0) ? vec.x : elu_alpha * (expf(vec.x) - 1);
            result.y = (vec.y > 0) ? vec.y : elu_alpha * (expf(vec.y) - 1);
            result.z = (vec.z > 0) ? vec.z : elu_alpha * (expf(vec.z) - 1);
            result.w = (vec.w > 0) ? vec.w : elu_alpha * (expf(vec.w) - 1);
            
            // Vectorized store
            *reinterpret_cast<float4*>(out + tid_base) = result;
        } else {
            // Scalar tail handling
            for (int i = 0; i < remaining; ++i) {
                const float val = x[tid_base + i];
                out[tid_base + i] = (val > 0) ? val : elu_alpha * (expf(val) - 1);
            }
        }
    }
}

torch::Tensor elu_constant_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    const int n = x.numel();

    // Copy alpha value to constant memory
    cudaMemcpyToSymbol(elu_alpha, &alpha, sizeof(float));

    // Account for vector width (4 elements) in grid calculation
    constexpr int vec_size = 4;
    const int threads = 512;
    const int blocks = (n + (threads * vec_size) - 1) / (threads * vec_size);

    // Use CUDA 1D grid/block addressing
    elu_kernel_constant<<<blocks, threads>>>(x.data_ptr<float>(),
                                             out.data_ptr<float>(),
                                             n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_constant_cuda, "Constant memory ELU activation (CUDA)");
}
