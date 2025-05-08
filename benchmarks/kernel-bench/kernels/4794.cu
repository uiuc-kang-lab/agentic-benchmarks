#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__constant__ float d_norm;

__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 4) + tid;
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (idx < numel) {
            float val = input[idx];
            sum1 += val * val;
        }
        idx += blockDim.x;
    }
    
    shared_sum[tid] = sum1 + sum2 + sum3 + sum4;
    __syncthreads();

    // Unrolled reduction
    if (tid < 128) shared_sum[tid] += shared_sum[tid + 128]; __syncthreads();
    if (tid < 64) shared_sum[tid] += shared_sum[tid + 64]; __syncthreads();
    if (tid < 32) {
        shared_sum[tid] += shared_sum[tid + 32];
        __syncwarp();
        shared_sum[tid] += shared_sum[tid + 16];
        __syncwarp();
        shared_sum[tid] += shared_sum[tid + 8];
        __syncwarp();
        shared_sum[tid] += shared_sum[tid + 4];
        __syncwarp();
        shared_sum[tid] += shared_sum[tid + 2];
        __syncwarp();
        shared_sum[tid] += shared_sum[tid + 1];
    }
    
    if (tid == 0) {
        atomicAdd(norm_out, shared_sum[0]);
    }
}

__global__ void normalize_kernel(const float* input, float* output, int numel) {
    const int idx = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int element_idx = idx + i * blockDim.x;
        if (element_idx < numel) {
            output[element_idx] = input[element_idx] / d_norm;
        }
    }
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
    const int threads = 256;
    const int blocks = (numel + threads * 4 - 1) / (threads * 4);

    compute_norm_kernel<<<min(blocks, 65535), threads>>>(input_ptr, norm_ptr, numel);

    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrtf(norm_val + 1e-8f);
    cudaMemcpyToSymbol(d_norm, &norm_val, sizeof(float));

    normalize_kernel<<<(numel + threads * 4 - 1) / (threads * 4), threads>>>(input_ptr, output_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm with unrolled loops");
}