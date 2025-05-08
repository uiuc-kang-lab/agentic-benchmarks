#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Kernel to compute the sum of squares using a grid-stride loop with optimized indexing
__global__ void compute_norm_kernel_indexing(const float* __restrict__ input, float* __restrict__ norm_out, int numel) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;

    // Each thread processes multiple elements using a grid-stride loop
    for (; idx < numel; idx += stride) {
        float val = input[idx];
        sum += val * val;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    if (tid < 128) {
        sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();

    if (tid < 64) {
        sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();

    // Final warp-level reduction without extra syncthreads
    if (tid < 32) {
        volatile float* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 32];
        vsdata[tid] += vsdata[tid + 16];
        vsdata[tid] += vsdata[tid + 8];
        vsdata[tid] += vsdata[tid + 4];
        vsdata[tid] += vsdata[tid + 2];
        vsdata[tid] += vsdata[tid + 1];
        if (tid == 0) {
            atomicAdd(norm_out, vsdata[0]);
        }
    }
}

// Kernel to normalize the tensor using the computed Frobenius norm
__global__ void normalize_kernel(const float* __restrict__ input, float* __restrict__ output, float norm, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        output[idx] = input[idx] / norm;
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

    // Clear the norm accumulator
    cudaMemset(norm_ptr, 0, sizeof(float));

    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);

    // Launch the kernel for sum of squares with optimized thread mapping
    compute_norm_kernel_indexing<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    // Retrieve the result and compute the Frobenius norm
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    // Launch the normalization kernel using grid-stride loop
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization with optimized thread/block indexing");
}
