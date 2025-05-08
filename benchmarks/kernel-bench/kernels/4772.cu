#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Optimized CUDA kernel for computing sum of squares using vectorized loads
__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Process main part using 128-bit loads (float4), assuming input is 16-byte aligned
    int n_vec = numel / 4;  // number of float4 elements
    for (int i = index; i < n_vec; i += stride) {
        float4 v = __ldg(reinterpret_cast<const float4*>(input) + i);
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    
    // Process any remaining elements
    int rem_start = n_vec * 4;
    for (int i = rem_start + index; i < numel; i += stride) {
        float val = __ldg(input + i);
        sum += val * val;
    }

    // Optimized reduction using warp shuffle instructions
    unsigned int mask = 0xffffffff;
    // Perform warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Each warp writes its reduced sum to shared memory
    if ((tid & 31) == 0) {
        sdata[tid >> 5] = sum;
    }
    __syncthreads();

    // Final reduction by the first warp
    if (tid < (blockDim.x >> 5)) {
        sum = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (tid == 0) {
            atomicAdd(norm_out, sum);
        }
    }
}

// Optimized normalization kernel using vectorized loads and stores
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Process main part using vectorized operations
    int n_vec = numel / 4;  // number of float4's
    for (int i = tid; i < n_vec; i += stride) {
        float4 in_val = __ldg(reinterpret_cast<const float4*>(input) + i);
        float4 out_val;
        out_val.x = in_val.x / norm;
        out_val.y = in_val.y / norm;
        out_val.z = in_val.z / norm;
        out_val.w = in_val.w / norm;
        reinterpret_cast<float4*>(output)[i] = out_val;
    }

    // Process any remaining elements
    int rem_start = n_vec * 4;
    for (int i = rem_start + tid; i < numel; i += stride) {
        output[i] = __ldg(input + i) / norm;
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

    // Compute squared norm with vectorized loads
    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    // Normalize using vectorized load and store
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Frobenius norm normalization with vectorized operations");
}