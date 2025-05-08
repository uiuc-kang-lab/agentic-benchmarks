#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void stridedVectorMultiplyKernel(const float* __restrict__ A,
                                           float* __restrict__ C,
                                           float s,
                                           int64_t size)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vector_stride = stride * 4;
    
    // Process chunks of 4 elements per thread with stride
    for (int idx = tid * 4; idx < size - 3; idx += vector_stride) {
        float4 in_data = *reinterpret_cast<const float4*>(&A[idx]);
        float4 out_data;
        out_data.x = in_data.x * s;
        out_data.y = in_data.y * s;
        out_data.z = in_data.z * s;
        out_data.w = in_data.w * s;
        *reinterpret_cast<float4*>(&C[idx]) = out_data;
    }
    
    // Handle remaining elements
    const int remainder_start = ((size >> 2) << 2);  // Round down to nearest multiple of 4
    for (int idx = remainder_start + tid; idx < size; idx += stride) {
        C[idx] = A[idx] * s;
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    const int threads = 256;
    const int min_blocks_per_sm = 2;
    const int multiprocessor_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int blocks = min(256, multiprocessor_count * min_blocks_per_sm);
    
    stridedVectorMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                    C.data_ptr<float>(),
                                                    s,
                                                    size);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided vectorized matrix-scalar multiplication");
}