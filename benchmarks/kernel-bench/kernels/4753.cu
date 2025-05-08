#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Template device function to unroll the loop for computing partial sum of squares
template <int UNROLL_FACTOR>
__device__ inline float unroll_loop(const float* input, int numel, int idx, int stride) {
    float sum = 0.0f;
    // Calculate the number of iterations this thread will perform
    int n = (numel > idx) ? (numel - idx) / stride : 0;
    int unroll_iters = n / UNROLL_FACTOR;
    int remainder = n % UNROLL_FACTOR;
    int base = idx;

    #pragma unroll
    for (int i = 0; i < unroll_iters; i++) {
        sum += input[base] * input[base] +
               input[base + stride] * input[base + stride] +
               input[base + 2 * stride] * input[base + 2 * stride] +
               input[base + 3 * stride] * input[base + 3 * stride];
        base += stride * UNROLL_FACTOR;
    }
    for (int i = 0; i < remainder; i++) {
        sum += input[base] * input[base];
        base += stride;
    }
    return sum;
}

// CUDA kernel for computing sum-of-squares using manual loop unrolling
__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    if (idx < numel) {
        sum = unroll_loop<4>(input, numel, idx, stride);
    }
    sdata[tid] = sum;
    __syncthreads();

    // Intra-block reduction with unrolling
    #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(norm_out, sdata[0]);
    }
}

// CUDA kernel for normalizing the tensor
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / norm;
    }
}

// Forward function called from Python
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

    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Unrolled Frobenius norm normalization");
}
