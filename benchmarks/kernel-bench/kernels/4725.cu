#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Device function: warp-level reduction using shfl_down_sync
__device__ inline float warpReduceSum(float val) {
    // Unroll reduction within a warp
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Device function: block-level reduction using shared memory and warp reduction
__device__ inline float blockReduceSum(float *sdata, int tid) {
    // Reduction in shared memory
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();
    
    // For threads within a warp, use warp-level reduction
    if (tid < 32) {
        float val = sdata[tid];
        val = warpReduceSum(val);
        return val;
    }
    return 0.0f; // Only thread 0 in the warp will return a valid result
}

// Kernel to compute sum of squares (partial Frobenius norm) using modular reduction functions
__global__ void compute_norm_kernel_modular(const float* input, float* norm_out, int numel) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Accumulate partial sum of squares
    while (idx < numel) {
        float val = input[idx];
        sum += val * val;
        idx += blockDim.x * gridDim.x;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Use the modular block reduction function
    float blockSum = blockReduceSum(sdata, tid);

    // The first thread of each block atomically adds the block result to global memory
    if (tid == 0) {
        atomicAdd(norm_out, blockSum);
    }
}

// Kernel for normalizing the input tensor using the computed Frobenius norm
__global__ void normalize_kernel_modular(const float* input, float* output, float norm, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / norm;
    }
}

// Host function callable from PyTorch
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    // Create output tensor and a tensor for the norm (initialized to zero)
    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();

    int numel = input.numel();
    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);

    // Ensure norm_tensor is zero
    cudaMemset(norm_ptr, 0, sizeof(float));

    // Launch the modular reduction kernel to compute the sum of squares
    compute_norm_kernel_modular<<<blocks, threads>>>(input_ptr, norm_ptr, numel);
    
    // Retrieve the sum of squares and compute the Frobenius norm
    float norm_sum;
    cudaMemcpy(&norm_sum, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_sum = sqrt(norm_sum);

    // Launch normalization kernel using the computed norm
    normalize_kernel_modular<<<blocks, threads>>>(input_ptr, output_ptr, norm_sum, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization (modular device functions)");
}
