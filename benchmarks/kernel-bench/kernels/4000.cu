#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Aligned structure for coalesced memory access
struct alignas(16) AlignedFloat4 {
    float x, y, z, w;
};

__global__ void elu_kernel_aligned(const AlignedFloat4* __restrict__ input,
                                 AlignedFloat4* __restrict__ output,
                                 float alpha,
                                 int n_aligned) {
    // Calculate aligned index for coalesced access
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_aligned) {
        // Load aligned data
        AlignedFloat4 in = input[idx];
        AlignedFloat4 out;
        
        // Process data maintaining alignment
        out.x = (in.x > 0) ? in.x : alpha * (expf(in.x) - 1);
        out.y = (in.y > 0) ? in.y : alpha * (expf(in.y) - 1);
        out.z = (in.z > 0) ? in.z : alpha * (expf(in.z) - 1);
        out.w = (in.w > 0) ? in.w : alpha * (expf(in.w) - 1);
        
        // Store aligned result
        output[idx] = out;
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    
    int n = x.numel();
    int n_aligned = n / 4;  // Number of aligned float4 elements
    
    // Optimize thread block size for alignment
    const int threads = 128;  // Multiple of warp size (32)
    const int blocks = (n_aligned + threads - 1) / threads;
    
    elu_kernel_aligned<<<blocks, threads>>>(
        reinterpret_cast<const AlignedFloat4*>(x.data_ptr<float>()),
        reinterpret_cast<AlignedFloat4*>(out.data_ptr<float>()),
        alpha,
        n_aligned
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda, "ELU activation (CUDA)");
}