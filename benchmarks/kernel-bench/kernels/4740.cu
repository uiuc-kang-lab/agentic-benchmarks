#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA kernel for computing sum of squares with stride loop handling
__global__ void compute_norm_kernel_stride(const float* input, float* norm_out, int numel) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Use stride loops to process elements beyond the initial thread workload
    for (int i = idx; i < numel; i += blockDim.x * gridDim.x) {
        float val = input[i];
        sum += val * val;
    }

    // Write partial sum to shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Only one thread per block writes the block's result
    if (tid == 0) {
        atomicAdd(norm_out, sdata[0]);
    }
}

// CUDA kernel for tensor normalization
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < numel; i += blockDim.x * gridDim.x) {
        output[i] = input[i] / norm;
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
    const int blocks = min(65535, (numel + threads - 1) / threads);

    // Launch kernel to compute the sum of squares with stride loop handling
    compute_norm_kernel_stride<<<blocks, threads>>>(input_ptr, norm_ptr, numel);
    
    // Retrieve the computed sum and calculate the Frobenius norm
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    // Launch kernel to normalize the tensor
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization");
}