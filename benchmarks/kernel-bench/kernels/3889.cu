#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that evenly partitions the input data among all threads
__global__ void softsign_kernel_balanced(const float* __restrict__ x, float* __restrict__ out, int num_elements) {
    int total_threads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute the number of elements each thread should process (ceiling division)
    int chunk = (num_elements + total_threads - 1) / total_threads;
    int start_idx = tid * chunk;
    int end_idx = min(start_idx + chunk, num_elements);
    
    for (int i = start_idx; i < end_idx; i++) {
        float val = x[i];
        out[i] = val / (1.0f + fabsf(val));
    }
}

// Host function wrapping the kernel
torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    // Choose 256 threads per block. The grid size is chosen to avoid underutilization
    int threads = 256;
    int blocks = std::min((num_elements + threads - 1) / threads, 1024);
    
    softsign_kernel_balanced<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_elements);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced Softsign activation (CUDA)");
}
