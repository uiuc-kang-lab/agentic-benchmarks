#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This function computes tanh element-wise for a float4 vector
// Using tanhf for single-precision floating point computation
template <typename scalar_t>
__device__ __forceinline__ float4 compute_tanh_vec4(const float4 in) {
    float4 out;
    out.x = tanhf(in.x);
    out.y = tanhf(in.y);
    out.z = tanhf(in.z);
    out.w = tanhf(in.w);
    return out;
}

// Kernel performing vectorized tanh activation without any unnecessary atomic operations
// Since each thread processes distinct data, no race conditions occur and atomics are not required
template <typename scalar_t>
__global__ void atomic_free_vectorized_tanh_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vec4_size = size / 4;

    // Reinterpret input/output pointers as float4 pointers for vectorized loads/stores
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);

    // Process vectorized portion in chunks of 4 elements
    for (int i = tid; i < vec4_size; i += stride) {
        float4 in_val = input_vec[i];
        output_vec[i] = compute_tanh_vec4<scalar_t>(in_val);
    }

    // Handle remaining elements that do not fit into a float4 block
    int leftover_start = vec4_size * 4;
    for (int i = leftover_start + tid; i < size; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

// Forward function exposing the kernel to Python via pybind11
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() / 4 + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "atomic_free_vectorized_tanh_kernel", ([&] {
        atomic_free_vectorized_tanh_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward vectorized without unnecessary atomics (CUDA)");
}
