#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 512

// Vectorized warps with simplified reduction
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float blockReduce(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduce(val);
    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    return warpReduce(val);
}

__global__ void optimized_norm_kernel(const float* __restrict__ input,
                                    float* __restrict__ norm_out,
                                    int numel) {
    float sum = 0.0f;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    // Optimized grid-stride loop with 2D access pattern
    unsigned int i = tid;
    for (; i + 3*stride < numel; i += stride*4) {
        float a = input[i];
        float b = input[i + stride];
        float c = input[i + 2*stride];
        float d = input[i + 3*stride];
        sum += a * a + b * b + c * c + d * d;
    }
    for (; i < numel; i += stride)
        sum += input[i] * input[i];

    sum = blockReduce(sum);
    if (threadIdx.x == 0)
        atomicAdd(norm_out, sum);
}

__constant__ float cached_norm;

__global__ void optimized_normalize_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int numel) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    // Batched normalization with optimized grid stride
    for (unsigned int i = tid; i < numel; i += stride)
        output[i] = input[i] / cached_norm;
}

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

    // Calculate grid dimensions using 2D pattern
    int threads = BLOCK_SIZE;
    int blocks = (numel + threads - 1) / threads;
    const dim3 grid_dim(std::min(blocks, 65535), (blocks + 65534) / 65535);

    optimized_norm_kernel<<<grid_dim, threads>>>(input_ptr, norm_ptr, numel);

    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrtf(norm_val);
    cudaMemcpyToSymbol(cached_norm, &norm_val, sizeof(float));

    optimized_normalize_kernel<<<grid_dim, threads>>>(input_ptr, output_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Frobenius normalization");
}