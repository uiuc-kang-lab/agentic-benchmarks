#include <torch/extension.h>

// CUDA kernel for reverse cumulative sum
__global__ void reverse_cumsum_kernel(float* x, float* out, int64_t size, int64_t dim_break) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32; // Each warp consists of 32 threads
    int lane_id = tid % 32; // Lane within a warp

    // Calculate position based on warp
    int pos = warp_id * 32 + lane_id;

    // Boundary check
    if (pos < size) {
        // Initialize a temporary sum holder
        float temp_sum = 0;
        
        // Iterate over elements in reverse within the same warp to reduce divergence
        for (int i = pos; i >= 0; i -= 32) {
            temp_sum += x[i];
        }

        // Store the result
        out[pos] = temp_sum;
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int device_id) {
    // Ensure the tensor is contiguous and on CUDA
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    at::Tensor out = torch::zeros_like(x);
    int64_t size = x.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Launch kernel
    reverse_cumsum_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, x.size(-1));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum along a specified dimension (CUDA)");
}