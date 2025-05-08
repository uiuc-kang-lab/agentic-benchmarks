#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

// Kernel to compute the sum of squares using vectorized loads for memory coalescing
__global__ void compute_norm_kernel(const float* __restrict__ input, float* norm_out, int numel) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    // Calculate how many float4 loads we can perform
    int vec_count = numel / 4;
    int scalar_start = vec_count * 4;

    // Global thread index and total stride
    int tid_global = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    // Use vectorized (float4) loads to ensure coalesced accesses
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    for (int i = tid_global; i < vec_count; i += stride) {
        float4 v = input_vec[i];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Process any remaining elements that don't fit into a float4 load
    for (int i = scalar_start + tid_global; i < numel; i += stride) {
        float v = input[i];
        sum += v * v;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Intra-block reduction
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

// Kernel to normalize the tensor using vectorized stores for memory coalescing
__global__ void normalize_kernel(const float* __restrict__ input, float* output, float norm, int numel) {
    int vec_count = numel / 4;
    int scalar_start = vec_count * 4;

    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);

    // Process the bulk of the data
    for (int i = tid_global; i < vec_count; i += stride) {
        float4 in_val = input_vec[i];
        float4 out_val;
        out_val.x = in_val.x / norm;
        out_val.y = in_val.y / norm;
        out_val.z = in_val.z / norm;
        out_val.w = in_val.w / norm;
        output_vec[i] = out_val;
    }

    // Process any remaining elements
    for (int i = scalar_start + tid_global; i < numel; i += stride) {
        output[i] = input[i] / norm;
    }
}

// Forward function exposed to Python
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

    // Reset norm to zero before kernel launch
    cudaMemset(norm_ptr, 0, sizeof(float));

    // Launch kernel to compute the sum of squares with coalesced memory accesses
    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    // Launch kernel to normalize the tensor using coalesced vectorized stores
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Frobenius norm normalization");
}
