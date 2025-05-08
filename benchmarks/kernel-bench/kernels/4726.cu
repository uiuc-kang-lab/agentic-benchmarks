#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Kernel to compute the sum of squares using shared memory tiling
// Each thread performs a grid-stride loop and accumulates its partial sum
__global__ void compute_norm_kernel_shared(const float* input, float* norm_out, int numel) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;

    // Grid-stride loop: each thread processes multiple elements
    for (int i = blockIdx.x * blockDim.x + tid; i < numel; i += blockDim.x * gridDim.x) {
        sum += input[i] * input[i];
    }

    // Store the partial sum in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Atomic add to global memory to accumulate the block results
    if (tid == 0) {
        atomicAdd(norm_out, sdata[0]);
    }
}

// Kernel to finalize the norm by computing the square root on the GPU
// This avoids an extra host-device transfer for a single float value
__global__ void finalize_norm(float* norm) {
    norm[0] = sqrtf(norm[0]);
}

// Normalization kernel that leverages shared memory to cache the computed norm
// The norm is loaded once per block into shared memory to reduce repeated global memory accesses
__global__ void normalize_kernel_shared(const float* input, float* output, float* norm_global, int numel) {
    __shared__ float norm;
    if (threadIdx.x == 0) {
        norm = norm_global[0];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / norm;
    }
}

// Host function exposed to PyTorch
// Performs input validation, launches the kernels, and returns the normalized tensor
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    // Create output tensor and a tensor to hold the intermediate norm value
    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();

    int numel = input.numel();
    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);

    // Kernel 1: Compute the sum of squares using shared memory
    compute_norm_kernel_shared<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    // Kernel 2: Finalize the norm by taking the square root (done on GPU to avoid extra cudaMemcpy overhead)
    finalize_norm<<<1, 1>>>(norm_ptr);

    // Kernel 3: Normalize the tensor; each block loads the norm value into shared memory
    normalize_kernel_shared<<<blocks, threads>>>(input_ptr, output_ptr, norm_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization with shared memory usage");
}
