#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Optimized kernel combining vectorized loads with efficient shared memory usage
__global__ void compute_and_normalize_kernel(const float* input, float* output, float* norm_out, int numel) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Process main part using 128-bit loads (float4), assuming input is 16-byte aligned
    int n_vec = numel / 4;  // number of float4 elements
    for (int i = index; i < n_vec; i += stride) {
        // Use __ldg() for read-only access
        float4 v = __ldg(reinterpret_cast<const float4*>(input) + i);
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    
    // Process any remaining elements
    int rem_start = n_vec * 4;
    for (int i = rem_start + index; i < numel; i += stride) {
        float val = __ldg(input + i);
        sum += val * val;
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

    __syncthreads();

    // Calculate norm once per block after reduction
    if (tid == 0) {
        float norm_val;
        cudaMemcpy(&norm_val, norm_out, sizeof(float), cudaMemcpyDeviceToHost);
        norm_val = sqrt(norm_val);
        atomicExch(norm_out, norm_val);
    }

    __syncthreads();

    // Normalize using the calculated norm
    for (int i = index; i < numel; i += stride) {
        output[i] = input[i] / (*norm_out);
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

    // Combined kernel to compute norm and normalize
    compute_and_normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Frobenius norm normalization");
}