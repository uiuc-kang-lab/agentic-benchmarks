#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Bulk kernel for processing float data in groups of 4 without internal divergence
__global__ void tanh_bulk_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int vec_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread processes a complete float4 group; grid is sized to cover vec_count groups
    if (idx < vec_count) {
        float4 in_val = reinterpret_cast<const float4*>(input)[idx];
        float4 out_val;
        out_val.x = tanhf(in_val.x);
        out_val.y = tanhf(in_val.y);
        out_val.z = tanhf(in_val.z);
        out_val.w = tanhf(in_val.w);
        reinterpret_cast<float4*>(output)[idx] = out_val;
    }
}

// Tail kernel for processing the remaining float elements with an exact thread count
__global__ void tanh_tail_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int start,
                                  int num_remain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_remain) {
        int g_idx = start + idx;
        output[g_idx] = tanhf(input[g_idx]);
    }
}

// Generic kernel for non-float types using a uniform strided loop
// All threads iterate over the same number of loop iterations to reduce divergence
template <typename scalar_t>
__global__ void tanh_generic_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      int size,
                                      int total_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_iters = (size + total_threads - 1) / total_threads;
    for (int i = 0; i < n_iters; i++) {
        int idx = tid + i * total_threads;
        if (idx < size) {
            output[idx] = tanh(input[idx]);
        }
    }
}

// Forward function: splits the work for float type into two kernels (bulk and tail) to avoid
// divergent branches within warps. For other types, a uniform strided loop is used.
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int numel = input.numel();

    if (input.scalar_type() == at::ScalarType::Float) {
        // Process floats in vectorized groups of 4
        int vec_count = numel / 4;  // number of complete float4 groups
        int remainder = numel % 4;  // remaining elements
        if (vec_count > 0) {
            int threads = 256;
            int blocks = (vec_count + threads - 1) / threads;
            tanh_bulk_kernel<<<blocks, threads>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                vec_count
            );
        }
        if (remainder > 0) {
            int start = vec_count * 4;
            int threads = remainder;  // launch exactly the needed number of threads
            int blocks = 1;         // remainder is small (0 to 3)
            tanh_tail_kernel<<<blocks, threads>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                start,
                remainder
            );
        }
    } else {
        // Fallback for other floating types with a uniform stride loop
        const int threads = 256;
        int blocks = (numel + threads - 1) / threads;
        int total_threads = blocks * threads;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_generic_kernel", ([&] {
            tanh_generic_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                numel,
                total_threads
            );
        }));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with minimized warp divergence (CUDA)");
}
