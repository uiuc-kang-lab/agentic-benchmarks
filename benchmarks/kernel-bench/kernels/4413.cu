#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Warp-aligned reduction without conditionals
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduction optimized for warp alignment
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float warp_sums[32];
    const int lid = threadIdx.x % warpSize;
    const int wid = threadIdx.x / warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    
    // Warp-level reduction
    val = warpReduceSum(val);
    
    // Write reduced warp values to shared memory
    if (lid == 0) {
        warp_sums[wid] = val;
    }
    __syncthreads();
    
    // First warp reduces all warp sums
    if (wid == 0) {
        val = (lid < num_warps) ? warp_sums[lid] : 0.0f;
        val = warpReduceSum(val);
    }
    return val;
}

__global__ void instance_norm_kernel_aligned(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int N,
    const int C,
    const int H,
    const int W,
    const float eps
) {
    const int instance_idx = blockIdx.x;
    const int n = instance_idx / C;
    const int c = instance_idx % C;
    const int HW = H * W;
    
    // Align spatial dimension to warp size
    const int aligned_HW = ((HW + warpSize - 1) / warpSize) * warpSize;
    
    // Input/output pointers for current instance
    const float* x_ptr = input + (n * C + c) * HW;
    float* y_ptr = output + (n * C + c) * HW;
    
    // Compute sums using warp-aligned accesses
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    #pragma unroll 4
    for (int idx = threadIdx.x; idx < aligned_HW; idx += blockDim.x) {
        const bool valid = idx < HW;
        const float val = valid ? x_ptr[idx] : 0.0f;
        sum += val;
        sum_sq += val * val;
    }
    
    // Reduce sums across block
    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);
    
    // Compute mean and variance
    __shared__ float mean_val, inv_std;
    if (threadIdx.x == 0) {
        mean_val = sum / HW;
        const float var = fmaxf(sum_sq / HW - mean_val * mean_val, 0.0f);
        inv_std = rsqrtf(var + eps);
    }
    __syncthreads();
    
    // Load scaling factors
    const float scale = weight ? weight[c] : 1.0f;
    const float shift = bias ? bias[c] : 0.0f;
    
    // Normalize with vectorized loads/stores where possible
    const int vector_size = 4;
    const int vector_elements = HW / vector_size;
    const float4* x_vec = reinterpret_cast<const float4*>(x_ptr);
    float4* y_vec = reinterpret_cast<float4*>(y_ptr);
    
    // Vector processing
    #pragma unroll 2
    for (int idx = threadIdx.x; idx < vector_elements; idx += blockDim.x) {
        float4 in = x_vec[idx];
        float4 out;
        out.x = (in.x - mean_val) * inv_std * scale + shift;
        out.y = (in.y - mean_val) * inv_std * scale + shift;
        out.z = (in.z - mean_val) * inv_std * scale + shift;
        out.w = (in.w - mean_val) * inv_std * scale + shift;
        y_vec[idx] = out;
    }
    
    // Handle remaining elements
    const int remaining_start = vector_elements * vector_size;
    #pragma unroll
    for (int idx = remaining_start + threadIdx.x; idx < HW; idx += blockDim.x) {
        const float normalized = (x_ptr[idx] - mean_val) * inv_std;
        y_ptr[idx] = normalized * scale + shift;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    
    const auto sizes = x.sizes();
    const int N = sizes[0];
    const int C = sizes[1];
    const int H = sizes[2];
    const int W = sizes[3];
    
    auto output = torch::empty_like(x);
    
    // Choose block size to maximize occupancy while maintaining warp alignment
    const int HW = H * W;
    const int block_size = (HW < 512) ? 128 : 256;
    
    const dim3 blocks(N * C);
    const dim3 threads(block_size);
    
    instance_norm_kernel_aligned<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps)
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CUDA)");
}