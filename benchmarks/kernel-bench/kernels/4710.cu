#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for computing sum of squares with minimal synchronization
__global__ void compute_norm_kernel_min_sync(const float* input, float* norm_out, int numel) {
    __shared__ float shared_sum[256];
    const unsigned int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute partial sums directly into registers
    float sum = 0.0f;
    while (idx < numel) {
        float val = input[idx];
        sum += val * val;
        idx += blockDim.x * gridDim.x;
    }
    
    // Store to shared memory
    shared_sum[tid] = sum;
    
    // Single sync point before reduction
    __syncthreads();
    
    // Reduce within block - unrolled for efficiency
    if (tid < 128) shared_sum[tid] += shared_sum[tid + 128];
    if (tid < 64) shared_sum[tid] += shared_sum[tid + 64];
    
    // Warp-level reduction - no sync needed
    if (tid < 32) {
        volatile float* smem = shared_sum;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
        
        if (tid == 0) {
            atomicAdd(norm_out, smem[0]);
        }
    }
}

// CUDA kernel for normalization - vectorized loads/stores
__global__ void normalize_kernel_vec4(const float* input, float* output, 
                                    float norm, int numel) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx < numel) {
        float4 in4 = *reinterpret_cast<const float4*>(input + idx);
        float4 out4;
        out4.x = in4.x / norm;
        out4.y = in4.y / norm;
        out4.z = in4.z / norm;
        out4.w = in4.w / norm;
        *reinterpret_cast<float4*>(output + idx) = out4;
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
    const int blocks = min(65535, (numel + threads - 1) / threads);

    compute_norm_kernel_min_sync<<<blocks, threads>>>(input_ptr, norm_ptr, numel);
    
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);
    
    // Adjust block count for vectorized loads
    const int vec_blocks = min(65535, (numel/4 + threads - 1) / threads);
    normalize_kernel_vec4<<<vec_blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization");
}