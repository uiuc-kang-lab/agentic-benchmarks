#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Store constants for sigmoid computation in constant memory
__constant__ float sigmoid_constants[2] = {1.0f, 0.0f};

// Device function for computing sigmoid using constants
__device__ __forceinline__ float sigmoid_device(float x) {
    return sigmoid_constants[0] / (sigmoid_constants[0] + expf(-x));
}

// Vectorized kernel for float using float4 for coalesced memory accesses
__global__ void vectorized_sigmoid_kernel_float(const float* __restrict__ input,
                                                 float* __restrict__ output,
                                                 const int64_t size) {
    int n_vec = size / 4;  // number of complete float4 chunks
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = vec_idx; i < n_vec; i += stride) {
        float4 in_val = reinterpret_cast<const float4*>(input)[i];
        float4 out_val;
        out_val.x = sigmoid_device(in_val.x);
        out_val.y = sigmoid_device(in_val.y);
        out_val.z = sigmoid_device(in_val.z);
        out_val.w = sigmoid_device(in_val.w);
        reinterpret_cast<float4*>(output)[i] = out_val;
    }

    int tail_start = n_vec * 4;
    for (int i = tail_start + vec_idx; i < size; i += stride) {
        float in_val = input[i];
        output[i] = sigmoid_device(in_val);
    }
}

// Fallback scalar kernel for non-float types
template <typename scalar_t>
__global__ void sigmoid_kernel_scalar(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      const int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        float val = static_cast<float>(input[i]);
        float r = sigmoid_constants[0] / (sigmoid_constants[0] + expf(-val));
        output[i] = static_cast<scalar_t>(r);
    }
}

// Host API
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    const int threads = 256;

    if (input.scalar_type() == at::ScalarType::Float && size >= 4) {
        int n_vec = size / 4;
        int blocks = (n_vec + threads - 1) / threads;
        vectorized_sigmoid_kernel_float<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    } else {
        int blocks = (size + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel_scalar", ([&] {
            sigmoid_kernel_scalar<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                size
            );
        }));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA) with constant memory optimization");
}