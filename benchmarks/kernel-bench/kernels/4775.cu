#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Device function to compute the partial sum of squares for each thread using strided access
__device__ inline float compute_partial_sum(const float* input, int numel, int idx, int stride) {
    float sum = 0.0f;
    for (int i = idx; i < numel; i += stride) {
        float val = __ldg(&input[i]);  // Use __ldg for optimized cached loads
        sum += val * val;
    }
    return sum;
}

// CUDA kernel for computing sum of squares
__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    __shared__ float shared_sum[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Each thread computes its partial sum
    float sum = compute_partial_sum(input, numel, idx, stride);
    shared_sum[tid] = sum;
    __syncthreads();

    // Intra-block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(norm_out, shared_sum[0]);
    }
}

// CUDA kernel for normalizing the tensor
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float in_val = __ldg(&input[idx]);  // Use __ldg for optimized cached loads
        output[idx] = in_val / norm;
    }
}

// C++ forward function called from Python
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
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    // Compute sum of squares using the modular kernel
    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    // Copy the computed norm sum from device to host and compute the square root
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    // Normalize the tensor
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization with aligned memory access");
}