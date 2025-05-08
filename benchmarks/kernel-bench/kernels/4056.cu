#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

template<typename T>
__device__ __forceinline__ T elu_op(T x, float alpha) {
    return (x > 0) ? x : alpha * (expf(x) - 1);
}

__global__ void optimized_elu_kernel(const float* __restrict__ x, 
                                   float* __restrict__ out,
                                   float alpha, 
                                   int n) {
    __shared__ float s_data[BLOCK_SIZE * ELEMENTS_PER_THREAD];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + tid;
    
    if (gid + 3 * BLOCK_SIZE < n) {
        float4 in = reinterpret_cast<const float4*>(x)[gid/4];
        
        s_data[tid] = elu_op(in.x, alpha);
        s_data[tid + BLOCK_SIZE] = elu_op(in.y, alpha);
        s_data[tid + 2*BLOCK_SIZE] = elu_op(in.z, alpha);
        s_data[tid + 3*BLOCK_SIZE] = elu_op(in.w, alpha);
    } else {
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int idx = gid + i * BLOCK_SIZE;
            if (idx < n) {
                s_data[tid + i*BLOCK_SIZE] = elu_op(x[idx], alpha);
            }
        }
    }
    
    __syncthreads();
    
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = gid + i * BLOCK_SIZE;
        if (idx < n) {
            out[idx] = s_data[tid + i*BLOCK_SIZE];
        }
    }
}

torch::Tensor optimized_elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = BLOCK_SIZE;
    const int blocks = (n + (threads * ELEMENTS_PER_THREAD) - 1) / (threads * ELEMENTS_PER_THREAD);
    
    optimized_elu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        alpha,
        n
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_elu_cuda, "Optimized ELU activation (CUDA)");
}