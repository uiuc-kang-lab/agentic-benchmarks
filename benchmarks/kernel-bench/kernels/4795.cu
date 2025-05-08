#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Declare a constant memory variable to hold the norm value
__constant__ float d_norm;

// CUDA kernel for computing the sum of squares for Frobenius norm
__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_sum[tid] = 0.0f;
    
    // Accumulate partial sums
    while (idx < numel) {
        float val = input[idx];
        shared_sum[tid] += val * val;
        idx += blockDim.x * gridDim.x;
    }
    __syncthreads();

    // Intra-block reduction using warp-level primitives first
    // Warp reduction first (no sync needed within a warp)
    if (blockDim.x >= 64) {
        shared_sum[tid] += shared_sum[tid + 32];
    }
    if (blockDim.x >= 32) {
        shared_sum[tid] += shared_sum[tid + 16];
    }
    shared_sum[tid] += shared_sum[tid + 8];
    shared_sum[tid] += shared_sum[tid + 4];
    shared_sum[tid] += shared_sum[tid + 2];
    shared_sum[tid] += shared_sum[tid + 1];
    
    // Final reduction across warps
    if (blockDim.x > 32) {
        __syncthreads();
        for (int stride = blockDim.x / 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_sum[tid] += shared_sum[tid + stride];
            }
            __syncthreads();
        }
    }
    
    // Use atomic addition to update global sum
    if (tid == 0) {
        atomicAdd(norm_out, shared_sum[0]);
    }
}

// CUDA kernel for normalizing the tensor using the precomputed norm stored in constant memory
__global__ void normalize_kernel(const float* input, float* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / d_norm;
    }
}

// Host function that launches the kernels
torch::Tensor forward(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    // Allocate output tensor and a tensor for accumulating the norm
    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());

    // Get raw pointers
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();

    int numel = input.numel();
    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);

    // Launch kernel to compute the sum of squares
    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    // Copy the accumulated value from device and compute the square root to get the Frobenius norm
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrtf(norm_val);

    // Copy the computed norm into constant memory
    cudaMemcpyToSymbol(d_norm, &norm_val, sizeof(float), 0, cudaMemcpyHostToDevice);

    // Launch kernel to normalize the input tensor
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization using constant memory for the norm value");
}
