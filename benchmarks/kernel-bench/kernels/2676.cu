#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE/WARP_SIZE)

__global__ void leaky_relu_kernel_warp(const float* __restrict__ x, 
                                     float* __restrict__ out,
                                     float negative_slope, 
                                     int n) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;  // warp ID
    int lane = tid % WARP_SIZE; // lane within the warp
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Load data into shared memory with vectorized loads when possible
    if (gid < n) {
        shared_data[tid] = x[gid];
    }
    __syncthreads();
    
    // Process elements within each warp
    if (gid < n) {
        float val = shared_data[tid];
        
        // Use warp-level primitives for faster communication
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            float val_next = __shfl_down_sync(0xffffffff, val, offset);
            if (lane < offset && gid + offset < n) {
                val = val > 0 ? val : val * negative_slope;
                val_next = val_next > 0 ? val_next : val_next * negative_slope;
            }
        }
        
        // Write result
        out[gid] = val > 0 ? val : val * negative_slope;
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = BLOCK_SIZE;
    const int blocks = (n + threads - 1) / threads;
    
    leaky_relu_kernel_warp<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        negative_slope,
        n
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward with warp optimization (CUDA)");
}