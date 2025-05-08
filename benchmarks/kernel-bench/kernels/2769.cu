#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <int VEC_SIZE>
__global__ void leaky_relu_warp_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    float alpha,
    int total_elements) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_tid = tid * VEC_SIZE;
    const int vec_size = VEC_SIZE * blockDim.x * gridDim.x;
    
    for(int i = vec_tid; i < total_elements; i += vec_size) {
        float local_vec[VEC_SIZE];
        
        #pragma unroll
        for(int v = 0; v < VEC_SIZE; v++) {
            if(i + v * blockDim.x < total_elements) {
                local_vec[v] = __ldg(&input[i + v * blockDim.x]);
            }
        }

        #pragma unroll
        for(int v = 0; v < VEC_SIZE; v++) {
            if(i + v * blockDim.x < total_elements) {
                float val = local_vec[v];
                output[i + v * blockDim.x] = (val > 0.0f ? val : __fmaf_rn(val, alpha, 0.0f));
            }
        }
    }

    // Warp-level boundary check and sync
    const int remaining = total_elements - vec_tid;
    const int valid_lanes = __ballot_sync(0xffffffff, remaining > 0);
    
    if(valid_lanes == 0) return;
    
    #pragma unroll
    for(int v = VEC_SIZE; v < 4; v++) {
        int elem_idx = vec_tid + v * blockDim.x;
        if(elem_idx < total_elements) {
            float val = __ldg(&input[elem_idx]);
            output[elem_idx] = val > 0.0f ? val : val * alpha;
        }
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int64_t n = x.numel();
    
    constexpr int VEC_SIZE = 4;
    const int threads = 128; // Optimal for H100's 32 warps per SM
    const int blocks = (n + threads * VEC_SIZE - 1) / (threads * VEC_SIZE);
    
    leaky_relu_warp_optimized<VEC_SIZE><<<blocks, threads>>>(x.data_ptr<float>(),
        out.data_ptr<float>(), negative_slope, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward with warp-coalesced vectorization");
}