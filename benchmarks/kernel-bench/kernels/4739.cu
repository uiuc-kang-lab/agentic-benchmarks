#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// CUDA kernel for computing sum of squares using __ldg and vectorized loads
__global__ void compute_norm_kernel_ldg(const float* __restrict__ input, float* norm_out, int numel) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    float sum = 0.0f;

    // Process the input in 128-bit (float4) chunks
    int vec_count = numel / 4;  // number of float4 groups
    int total_threads = blockDim.x * gridDim.x;
    int global_tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Treat input as float4 pointer for aligned 128-bit loads
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    for (int i = global_tid; i < vec_count; i += total_threads) {
        float4 v = __ldg(&input_vec[i]);
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Each thread stores its partial sum in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Perform parallel reduction in shared memory
    if (tid < 128) {
        sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
    if (tid < 64) {
        volatile float* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 64];
        float val = vsdata[tid];
        for (int offset = 32; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            atomicAdd(norm_out, val);
        }
    }

    // Process remaining elements that don't fit into a float4 (if any)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float rem_sum = 0.0f;
        int start = vec_count * 4;
        for (int i = start; i < numel; i++) {
            float v = __ldg(&input[i]);
            rem_sum += v * v;
        }
        atomicAdd(norm_out, rem_sum);
    }
}

// CUDA kernel for tensor normalization using __ldg for read-only access
__global__ void normalize_kernel_ldg(const float* __restrict__ input, float* output, float norm, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float val = __ldg(&input[idx]);
        output[idx] = val / norm;
    }
}

// Host function interfacing with PyTorch
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();
    int numel = input.numel();

    const int threads = 256;
    const int blocks = std::min(65535, (numel + threads - 1) / threads);

    // Launch kernel to compute the sum of squares with optimized __ldg and vectorized loads
    compute_norm_kernel_ldg<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    // Launch kernel to normalize the tensor
    normalize_kernel_ldg<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization with __ldg and vectorized loads");
}
