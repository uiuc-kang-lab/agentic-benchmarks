#include <torch/extension.h>

__device__ inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void swish_vectorized_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    extern __shared__ float buffer[];
    
    const int tid = threadIdx.x;
    const int vec_id = blockIdx.x * blockDim.x + tid;
    const int elements_per_thread = 4;
    const int tile_size = blockDim.x * elements_per_thread;
    
    float4* in_vec = reinterpret_cast<float4*>(buffer);
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* y_vec = reinterpret_cast<float4*>(y);

    for (int base = 0; base < n; base += gridDim.x * tile_size) {
        int load_pos = base + blockIdx.x * tile_size + tid;
        if (load_pos * 4 < n) {
            in_vec[tid] = x_vec[load_pos];
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < elements_per_thread; ++i) {
            int elem_idx = tid * elements_per_thread + i;
            if (base * 4 + elem_idx < n) {
                float val = buffer[elem_idx];
                buffer[elem_idx] = val * fast_sigmoid(val);
            }
        }

        int store_pos = base + blockIdx.x * tile_size + tid;
        if (store_pos * 4 < n) {
            y_vec[store_pos] = in_vec[tid];
        }
    }
}

torch::Tensor swish_vectorized_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    const int threads = 256;
    const int elements_per_block = threads * 4;
    const int blocks = (n + elements_per_block - 1) / elements_per_block;
    const size_t shared_mem = threads * sizeof(float4);
    
    swish_vectorized_kernel<<<blocks, threads, shared_mem>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_vectorized_forward, "Swish with vectorized shared memory (CUDA)");
}