#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Combined reduction kernel: computes the sum of squares (Frobenius norm squared) using minimal synchronizations
__global__ void compute_norm_kernel_combined(const float* input, float* norm_out, int numel) {
    // Assuming blockDim.x == 256
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Each thread computes a partial sum
    while (idx < numel) {
        float val = input[idx];
        sum += val * val;
        idx += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory in two stages
    if (tid < 128) {
        sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
    if (tid < 64) {
        sdata[tid] += sdata[tid + 64];
    }
    
    // Warp-level reduction (no explicit __syncthreads() needed here)
    if (tid < 32) {
        volatile float* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 32];
        vsdata[tid] += vsdata[tid + 16];
        vsdata[tid] += vsdata[tid + 8];
        vsdata[tid] += vsdata[tid + 4];
        vsdata[tid] += vsdata[tid + 2];
        vsdata[tid] += vsdata[tid + 1];
        if (tid == 0) {
            // Atomically accumulate the block sum into global memory
            atomicAdd(norm_out, vsdata[0]);
        }
    }
}

// Normalization kernel: divides each input element by the provided norm
__global__ void normalize_kernel_combined(const float* input, float* output, float norm, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / norm;
    }
}

// Host function interfacing with PyTorch
torch::Tensor forward(torch::Tensor input) {
    // Input validations
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    // Create output tensor and a tensor to hold the norm (initialized to zero)
    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());

    // Get raw pointers
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();

    int numel = input.numel();
    const int threads = 256;
    // Limit the number of blocks to avoid launching too many
    const int blocks = min(65535, (numel + threads - 1) / threads);

    // Ensure norm_tensor is zeroed out
    cudaMemset(norm_ptr, 0, sizeof(float));

    // First kernel: compute the sum of squares using combined reduction
    compute_norm_kernel_combined<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    // Retrieve the sum of squares from device and compute the Frobenius norm
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);
    
    // Second kernel: normalize the input tensor using the computed norm
    normalize_kernel_combined<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm tensor normalization with optimized reduction");
}
