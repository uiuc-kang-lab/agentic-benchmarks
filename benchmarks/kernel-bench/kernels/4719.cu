#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA kernel for computing sum of squares optimized
template <unsigned int blockSize> 
__global__ void compute_norm_kernel_optimized(const float* input, float* norm_out, int numel) {
    __shared__ float shared_sum[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockSize + tid;
    float sum = 0.0f;

    // Each thread computes a partial sum
    while (idx < numel) {
        float val = input[idx];
        sum += val * val;
        idx += blockSize * gridDim.x;
    }

    // Write to shared memory
    shared_sum[tid] = sum;
    __syncthreads();

    // Reduction within each block
    if (blockSize >= 512 && tid < 256) shared_sum[tid] += shared_sum[tid + 256]; __syncthreads();
    if (blockSize >= 256 && tid < 128) shared_sum[tid] += shared_sum[tid + 128]; __syncthreads();
    if (blockSize >= 128 && tid < 64) shared_sum[tid] += shared_sum[tid + 64]; __syncthreads();

    // Warp-level reduction
    if (tid < 32) {
        volatile float *vsmem = shared_sum; // Declare as volatile to prevent over-optimization
        if (blockSize >= 64) vsmem[tid] += vsmem[tid + 32];
        if (blockSize >= 32) vsmem[tid] += vsmem[tid + 16];
        if (blockSize >= 16) vsmem[tid] += vsmem[tid + 8];
        if (blockSize >= 8) vsmem[tid] += vsmem[tid + 4];
        if (blockSize >= 4) vsmem[tid] += vsmem[tid + 2];
        if (blockSize >= 2) vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) atomicAdd(norm_out, shared_sum[0]);
}

// CUDA kernel for tensor normalization
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
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

    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);

    compute_norm_kernel_optimized<256><<<blocks, threads>>>(input_ptr, norm_ptr, numel);
    
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