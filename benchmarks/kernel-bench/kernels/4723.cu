#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA kernel with minimized atomic operation for computing sum of squares
__global__ void compute_norm_kernel_atomic_minimized(const float* input, float* norm_out, int numel) {
    __shared__ float shared_sum[256];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Each thread computes partial sum
    while (idx < numel) {
        float val = input[idx];
        sum += val * val;
        idx += blockDim.x * gridDim.x;
    }

    // Write sum to shared memory
    shared_sum[tid] = sum;
    __syncthreads();

    // Reduce within block dynamically
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Only the first thread in each block adds result to the output
    if (tid == 0) {
        atomicAdd(norm_out, shared_sum[0]);
    }
}

// Kernel for normalizing the tensor
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / norm;
    }
}

// PyTorch interface
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
    const int blocks = min(65535, (numel + threads - 1) / threads);

    // Compute sum of squares with atomic operation minimized
    compute_norm_kernel_atomic_minimized<<<blocks, threads>>>(input_ptr, norm_ptr, numel);
    
    // Transfer and calculate norm value
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    // Launch normalization kernel
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization");
}