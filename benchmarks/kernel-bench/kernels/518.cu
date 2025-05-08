#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void multiplyKernelCoalesced(const float4* __restrict__ A,
                                       float4* __restrict__ C,
                                       float s,
                                       int64_t size_in_float4)
{
    __shared__ float shared_s;
    if (threadIdx.x == 0) {
        shared_s = s;
    }
    __syncthreads();
    
    // Each thread processes consecutive float4 elements
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size_in_float4) {
        float4 a4 = __ldg(&A[tid]);
        
        // Multiply each component with the scalar
        a4.x *= shared_s;
        a4.y *= shared_s;
        a4.z *= shared_s;
        a4.w *= shared_s;
        
        // Store result in coalesced manner
        C[tid] = a4;
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t total_elements = A.numel();
    
    // Ensure alignment to float4 boundary
    TORCH_CHECK(total_elements % 4 == 0, "Input tensor size must be multiple of 4");
    int64_t size_in_float4 = total_elements / 4;
    
    const int threads = 256;
    const int blocks = (size_in_float4 + threads - 1) / threads; // Calculate the number of blocks needed to cover the entire data domain
    
    multiplyKernelCoalesced<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(A.data_ptr<float>()),
        reinterpret_cast<float4*>(C.data_ptr<float>()),
        s,
        size_in_float4
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced matrix-scalar multiplication kernel");
}