#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define maximum elements that can fit in constant memory (for float, 64KB max)
// 64KB / sizeof(float) = 16384 elements
#define MAX_CONSTANT_SIZE 16384

// Declare constant memory for float input
__constant__ float d_input_const[MAX_CONSTANT_SIZE];

// CUDA kernel that reads input from constant memory
__global__ void relu_kernel_const(float* __restrict__ output, const int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = d_input_const[idx];
        output[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

// CUDA kernel that reads input from global memory
__global__ void relu_kernel_global(float* __restrict__ output, const float* __restrict__ input, const int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

// PyTorch wrapper function
// This function checks if the input tensor (of type float) can fit into constant memory.
// If yes, it copies the input into constant memory and launches a kernel that reads from it,
// otherwise it falls back to a kernel that reads directly from global memory.

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    // Use constant memory optimization only for float tensors that fit within constant memory
    if (input.scalar_type() == torch::kFloat32) {
        float* output_ptr = output.data_ptr<float>();
        if (size <= MAX_CONSTANT_SIZE) {
            // Copy input data into constant memory
            cudaMemcpyToSymbol(d_input_const, input.data_ptr<float>(), size * sizeof(float));
            relu_kernel_const<<<blocks, threads>>>(output_ptr, size);
        } else {
            // Fallback: input too large for constant memory, use global memory kernel
            const float* input_ptr = input.data_ptr<float>();
            relu_kernel_global<<<blocks, threads>>>(output_ptr, input_ptr, size);
        }
    } else {
        // For other floating point types, use the generic global memory kernel
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_global", ([&] {
            relu_kernel_global<<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                size
            );
        }));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward using constant memory (CUDA)");
}
