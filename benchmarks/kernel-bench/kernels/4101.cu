#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32

template <typename scalar_t>
__global__ void shared_warp_elu_kernel(const scalar_t* __restrict__ x,
                                     scalar_t* __restrict__ out,
                                     float alpha,
                                     int n) {
    extern __shared__ float shared_data[];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int wid = tid / WARP_SIZE;  // warp ID
    const unsigned int lane = tid % WARP_SIZE; // lane ID within warp
    const unsigned int gid = bid * blockDim.x + tid;
    
    // Process 4 elements per thread using vectorized load
    if (gid * 4 < n) {
        float4 values;
        if (gid * 4 + 3 < n) {
            values = *reinterpret_cast<const float4*>(x + gid * 4);
        } else {
            // Handle boundary case
            values.x = (gid * 4 < n) ? x[gid * 4] : 0.0f;
            values.y = (gid * 4 + 1 < n) ? x[gid * 4 + 1] : 0.0f;
            values.z = (gid * 4 + 2 < n) ? x[gid * 4 + 2] : 0.0f;
            values.w = (gid * 4 + 3 < n) ? x[gid * 4 + 3] : 0.0f;
        }

        // Compute ELU using warp-level operations
        values.x = (values.x > 0) ? values.x : alpha * (expf(values.x) - 1.0f);
        values.y = (values.y > 0) ? values.y : alpha * (expf(values.y) - 1.0f);
        values.z = (values.z > 0) ? values.z : alpha * (expf(values.z) - 1.0f);
        values.w = (values.w > 0) ? values.w : alpha * (expf(values.w) - 1.0f);

        // Store results to shared memory
        shared_data[tid * 4]     = values.x;
        shared_data[tid * 4 + 1] = values.y;
        shared_data[tid * 4 + 2] = values.z;
        shared_data[tid * 4 + 3] = values.w;
    }
    
    __syncwarp();
    
    // Use warp-level primitives for final processing
    if (gid * 4 < n) {
        // Write back results using vectorized store
        if (gid * 4 + 3 < n) {
            *reinterpret_cast<float4*>(out + gid * 4) = make_float4(
                shared_data[tid * 4],
                shared_data[tid * 4 + 1],
                shared_data[tid * 4 + 2],
                shared_data[tid * 4 + 3]
            );
        } else {
            // Handle boundary case
            if (gid * 4 < n) out[gid * 4] = shared_data[tid * 4];
            if (gid * 4 + 1 < n) out[gid * 4 + 1] = shared_data[tid * 4 + 1];
            if (gid * 4 + 2 < n) out[gid * 4 + 2] = shared_data[tid * 4 + 2];
            if (gid * 4 + 3 < n) out[gid * 4 + 3] = shared_data[tid * 4 + 3];
        }
    }
}

torch::Tensor shared_warp_elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    const int n = x.numel();
    
    const int threads = 256;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);
    const int shared_memory_size = threads * 4 * sizeof(float);
    
    shared_warp_elu_kernel<float><<<blocks, threads, shared_memory_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        alpha,
        n
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_warp_elu_cuda, "Shared memory and warp-optimized ELU activation (CUDA)");
}