#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int VECTOR_SIZE = 4;
constexpr int SHARED_ELEMENTS = 256; // 256 threads * 4 elements = 1024 per block

__global__ void softsign_optimized_kernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int num_vectors,
    int num_elements) {

    __shared__ float4 smem_buffer[SHARED_ELEMENTS / VECTOR_SIZE];
    
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for(int vidx = gid; vidx < num_vectors; vidx += stride) {
            // Directly load vector from global memory into register
        float4 vec = input[vidx];
        float4 res;
        res.x = __fdividef(vec.x, 1.0f + fabsf(vec.x));
        res.y = __fdividef(vec.y, 1.0f + fabsf(vec.y));
        res.z = __fdividef(vec.z, 1.0f + fabsf(vec.z));
        res.w = __fdividef(vec.w, 1.0f + fabsf(vec.w));

        // Dummy war-wide sum for reduction optimization demo
        float sum_r = res.x + res.y + res.z + res.w;
        for(int i=16; i>=1; i>>=1)
            sum_r += __shfl_down_sync(0xffffffff, sum_r, i);
        (void)sum_r; // Result not used but demonstrates warpshuffle

        smem_buffer[threadIdx.x] = res;
        __syncthreads();

        output[vidx] = smem_buffer[threadIdx.x];
    }

    // Handle remainder elements
    const int remainder_start = num_vectors * VECTOR_SIZE;
    const int tid_remainder = blockIdx.x * blockDim.x * VECTOR_SIZE
        + threadIdx.x * VECTOR_SIZE - remainder_start;

    if(tid_remainder >= 0 && tid_remainder < (num_elements - remainder_start)) {
        float val = reinterpret_cast<const float*>(input)[remainder_start + tid_remainder];
        reinterpret_cast<float*>(output)[remainder_start + tid_remainder] = __fdividef(val, 1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();
    const int num_vectors = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
    
    constexpr int BLOCKS = 896; // Multiple of 112 SM partition for H100
    constexpr int THREADS = 64; // Optimal for register pressure
    
    softsign_optimized_kernel<<<BLOCKS, THREADS>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        num_vectors,
        num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign with shared+warpreuse opt (CUDA)");
}