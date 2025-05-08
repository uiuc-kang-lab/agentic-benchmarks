/*
 * This CUDA extension computes the Frobenius norm of an input tensor and normalizes it.
 * It combines efficient reduction with warp-level primitives and vectorized memory accesses.
 *
 * The first kernel computes the sum of squares using minimal synchronization and an efficient reduction scheme.
 * The second kernel performs vectorized normalization using float4 loads/stores, with a fallback for tail elements.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <math.h>

// Optimized kernel to compute sum of squares (Frobenius norm squared).
// Uses unrolled reduction and warp-level shuffle for final reduction.
__global__ void compute_norm_kernel_efficient(const float* __restrict__ input, float* __restrict__ norm_out, int numel) {
    __shared__ float shared_sum[256];  // Assumes blockDim.x == 256
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float sum = 0.0f;
    // Strided access over the input for load balancing
    while (idx < numel) {
        float val = input[idx];
        sum += val * val;
        idx += blockDim.x * gridDim.x;
    }
    shared_sum[tid] = sum;
    __syncthreads();

    // Unrolled reduction in shared memory
    // Assumes blockDim.x == 256
    if (tid < 128) {
        shared_sum[tid] += shared_sum[tid + 128];
    }
    __syncthreads();

    if (tid < 64) {
        shared_sum[tid] += shared_sum[tid + 64];
    }
    __syncthreads();

    // Warp-level reduction using shuffle instructions
    if (tid < 32) {
        float warp_sum = shared_sum[tid];
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 16);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 8);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 4);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 2);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 1);
        if (tid == 0) {
            atomicAdd(norm_out, warp_sum);
        }
    }
}

// Optimized normalization kernel with vectorized loads/stores using float4
__global__ void normalize_kernel_efficient(const float* __restrict__ input, float* __restrict__ output, float norm, int numel) {
    // Each thread processes 4 elements at once
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;

    // Check if we can safely use vectorized loads/stores
    if (idx + 3 < numel) {
        float4 in4 = *reinterpret_cast<const float4*>(input + idx);
        float4 out4;
        out4.x = in4.x / norm;
        out4.y = in4.y / norm;
        out4.z = in4.z / norm;
        out4.w = in4.w / norm;
        *reinterpret_cast<float4*>(output + idx) = out4;
    } else {
        // Fallback for remaining elements
        for (int i = idx; i < numel; i++) {
            output[i] = input[i] / norm;
        }
    }
}

// The forward function that calls the two kernels in sequence
// 1. It computes the norm by summing squares
// 2. It retrieves the norm (taking the square root) on the CPU
// 3. It normalizes all elements using the computed norm

torch::Tensor forward(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    // Create output tensor and a tensor to hold the norm (initialized to zero)
    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());

    // Get raw pointers to the data
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();

    int numel = input.numel();

    // Launch parameters for the reduction kernel
    const int threads = 256;
    int blocks = std::min(65535, (numel + threads - 1) / threads);
    
    // Launch kernel to compute the sum of squares
    compute_norm_kernel_efficient<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    // Copy the computed sum from device to host and compute the square root
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    // Launch parameters for the vectorized normalization kernel
    // Each thread processes 4 elements.
    int vec_elements = (numel + 3) / 4;  // number of float4 groups needed
    int norm_blocks = std::min(65535, (vec_elements + threads - 1) / threads);
    normalize_kernel_efficient<<<norm_blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization (efficient version)");
}
