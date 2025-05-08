#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int VECTOR_SIZE = 4;
constexpr int BLOCK_SIZE = 256;

__global__ void vectorized_leaky_relu_kernel(const float* __restrict__ input, float* __restrict__ output, float slope, int num_elements) {
    extern __shared__ float4 shared_buffer[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vector_idx = tid * VECTOR_SIZE;
    int elements_per_block = blockDim.x * VECTOR_SIZE;
    
    // Load vectorized data into shared memory
    if (vector_idx < num_elements) {
        float4 val = *reinterpret_cast<const float4*>(&input[vector_idx]);
        shared_buffer[threadIdx.x] = val;
    }
    __syncthreads();

    // Process data from shared memory
    if (vector_idx < num_elements) {
        float4 loaded = shared_buffer[threadIdx.x];
        float4 result;
        result.x = fmaxf(loaded.x, loaded.x * slope);
        result.y = fmaxf(loaded.y, loaded.y * slope);
        result.z = fmaxf(loaded.z, loaded.z * slope);
        result.w = fmaxf(loaded.w, loaded.w * slope);
        *reinterpret_cast<float4*>(&output[vector_idx]) = result;
    }

    // Process remaining elements with grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x + gridDim.x * blockDim.x;
         i < (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
         i += gridDim.x * blockDim.x) {
        int vec_offset = i * VECTOR_SIZE;
        if (vec_offset < num_elements) {
            float4 val = *reinterpret_cast<const float4*>(&input[vec_offset]);
            float4 res;
            res.x = fmaxf(val.x, val.x * slope);
            res.y = fmaxf(val.y, val.y * slope);
            res.z = fmaxf(val.z, val.z * slope);
            res.w = fmaxf(val.w, val.w * slope);
            *reinterpret_cast<float4*>(&output[vec_offset]) = res;
        }
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int num_elements = x.numel();

    // Calculate vectorized parameters
    int vectorized_size = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
    dim3 blocks((vectorized_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    blocks.x = blocks.x < 65535 ? blocks.x : 65535;

    size_t shared_mem = BLOCK_SIZE * sizeof(float4);
    
    vectorized_leaky_relu_kernel<<<blocks, BLOCK_SIZE, shared_mem>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        negative_slope,
        num_elements
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "Vectorized LeakyReLU with shared memory");
}
