#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA kernel for computing sum of squares with optimized memory access
__global__ void compute_norm_kernel_aligned(const float* __restrict__ input, float* norm_out, int numel) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Use float4 for coalesced memory access when possible
    float4 inp4;
    int aligned_size = (numel / 4) * 4;
    unsigned int idx4 = idx * 4;
    
    // Process 4 elements at a time using float4
    while (idx4 < aligned_size) {
        inp4 = *reinterpret_cast<const float4*>(&input[idx4]);
        sum += inp4.x * inp4.x + inp4.y * inp4.y + 
               inp4.z * inp4.z + inp4.w * inp4.w;
        idx4 += blockDim.x * gridDim.x * 4;
    }
    
    // Handle remaining elements
    idx = aligned_size + tid;
    while (idx < numel) {
        float val = __ldg(&input[idx]);
        sum += val * val;
        idx += blockDim.x * gridDim.x;
    }

    sdata[tid] = sum;
    __syncthreads();

    if (tid < 128) {
        sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();

    if (tid < 64) {
        volatile float* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 64];
        float val = vsdata[tid];
        
        // Warp-level reduction
        for (int offset = 32; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        if (tid == 0) {
            atomicAdd(norm_out, val);
        }
    }
}

// CUDA kernel for normalization with aligned memory access
__global__ void normalize_kernel_aligned(const float* __restrict__ input, 
                                       float* output, float norm, int numel) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int aligned_size = (numel / 4) * 4;
    
    // Process 4 elements at a time using float4
    if (idx * 4 < aligned_size) {
        float4 inp4 = *reinterpret_cast<const float4*>(&input[idx * 4]);
        float4 out4;
        out4.x = inp4.x / norm;
        out4.y = inp4.y / norm;
        out4.z = inp4.z / norm;
        out4.w = inp4.w / norm;
        *reinterpret_cast<float4*>(&output[idx * 4]) = out4;
    }
    
    // Handle remaining elements
    idx = aligned_size + threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numel) {
        output[idx] = __ldg(&input[idx]) / norm;
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
    const int blocks = min(65535, (numel + threads * 4 - 1) / (threads * 4));

    compute_norm_kernel_aligned<<<blocks, threads>>>(input_ptr, norm_ptr, numel);
    
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    normalize_kernel_aligned<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization");
}