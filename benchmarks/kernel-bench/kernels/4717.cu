#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Optimized kernel to compute sum of squares using grid-stride loops,
// unrolled shared memory reduction, and warp-level intrinsic reduction
__global__ void compute_norm_kernel_combined(const float* __restrict__ input, float* norm_out, int numel) {
    __shared__ float shared_sum[256];
    const unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Grid-stride loop to accumulate partial sum of squares
    while (idx < numel) {
        float val = input[idx];
        sum += val * val;
        idx += blockDim.x * gridDim.x;
    }

    shared_sum[tid] = sum;
    __syncthreads();

    // Unrolled reduction: first reduce from 256 to 128 threads
    if (blockDim.x >= 256 && tid < 128) {
        shared_sum[tid] += shared_sum[tid + 128];
    }
    __syncthreads();

    // Reduce from 128 to 64 threads
    if (tid < 64) {
        shared_sum[tid] += shared_sum[tid + 64];
    }
    __syncthreads();

    // Warp-level reduction using __shfl_down_sync
    if (tid < 32) {
        float val = shared_sum[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            atomicAdd(norm_out, val);
        }
    }
}

// Normalization kernel that uses vectorized loads/stores with float4
// It processes a large portion of the tensor in a vectorized manner, with a fallback
// for any remaining elements not divisible by 4
__global__ void normalize_kernel_combined(const float* __restrict__ input, float* __restrict__ output,
                                           float norm, int numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_iters = numel / 4;  // number of complete float4 groups

    // Process vectorized portion
    for (int i = tid; i < vec_iters; i += gridDim.x * blockDim.x) {
        int offset = i * 4;
        float4 in_val = *reinterpret_cast<const float4*>(input + offset);
        float4 out_val;
        out_val.x = in_val.x / norm;
        out_val.y = in_val.y / norm;
        out_val.z = in_val.z / norm;
        out_val.w = in_val.w / norm;
        *reinterpret_cast<float4*>(output + offset) = out_val;
    }

    // Process the tail elements if numel is not divisible by 4
    int start = vec_iters * 4;
    for (int i = start + tid; i < numel; i += gridDim.x * blockDim.x) {
        output[i] = input[i] / norm;
    }
}

// Forward function that launches the two kernels sequentially
// First, it computes the Frobenius norm (sqrt of sum-of-squares) of the input tensor,
// then it normalizes the tensor using that norm.
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    // Create output tensor and tensor for the norm accumulation
    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();

    int numel = input.numel();
    const int threads = 256;
    const int blocks = std::min(65535, (numel + threads - 1) / threads);

    // Launch reduction kernel to compute sum of squares
    compute_norm_kernel_combined<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    // Retrieve the computed sum from device to host, then compute the square root
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrtf(norm_val);

    // Launch normalization kernel with vectorized memory accesses
    int vec_blocks = std::min(65535, ((numel / 4) + threads - 1) / threads);
    normalize_kernel_combined<<<vec_blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization with fused optimized reduction and vectorized normalization");
}
