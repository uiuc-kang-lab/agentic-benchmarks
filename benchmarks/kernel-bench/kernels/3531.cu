#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function to compute SELU activation for a single float
__device__ __forceinline__ float selu_activate(float x) {
    float res = (x > 0.0f) ? x : 1.67326324235437728481f * (expf(x) - 1.0f);
    return 1.05070098735548049342f * res;
}

// CUDA kernel that processes input in a vectorized manner for improved memory coalescing
__global__ void selu_kernel_mem_aligned(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          size_t numel) {
    // Total threads in the grid
    size_t total_threads = blockDim.x * gridDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process the bulk of data as float4 provided that input is contiguous and aligned
    size_t vectorized_elems = numel / 4; // number of float4 elements
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);

    // Grid-stride loop over vectorized portion
    for (size_t i = idx; i < vectorized_elems; i += total_threads) {
        float4 in_val = input_vec[i];
        float4 out_val;
        out_val.x = selu_activate(in_val.x);
        out_val.y = selu_activate(in_val.y);
        out_val.z = selu_activate(in_val.z);
        out_val.w = selu_activate(in_val.w);
        output_vec[i] = out_val;
    }

    // Handle remaining elements if numel is not divisible by 4
    size_t offset = vectorized_elems * 4;
    for (size_t i = idx; i < (numel - offset); i += total_threads) {
        output[offset + i] = selu_activate(input[offset + i]);
    }
}

// Host function launching the CUDA kernel
torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input tensor must be float32");

    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    
    // Using 256 threads, compute blocks based on vectorized elements
    const int threads = 256;
    size_t vectorized_elems = numel / 4;
    int blocks = (vectorized_elems + threads - 1) / threads;
    if(blocks == 0) blocks = 1;

    selu_kernel_mem_aligned<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Memory Aligned Access (CUDA)");
}
