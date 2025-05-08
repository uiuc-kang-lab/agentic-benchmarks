#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that evenly distributes work among threads by partitioning the total number of elements
// into contiguous chunks, ensuring each thread processes a balanced workload.
__global__ void softsign_kernel_even(const float* __restrict__ x, float* __restrict__ out, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    // Calculate the number of elements per thread (using ceiling division)
    int chunk = (num_elements + total_threads - 1) / total_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > num_elements) {
        end = num_elements;
    }
    for (int i = start; i < end; i++) {
        float val = x[i];
        out[i] = val / (1.0f + fabsf(val));
    }
}

// The forward function creates output tensor and launches the kernel
torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    // Choose a high thread count per block for maximum occupancy
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;
    
    softsign_kernel_even<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_elements);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with evenly distributed workload (CUDA)");
}
