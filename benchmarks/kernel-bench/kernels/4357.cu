#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

__global__ void instance_norm_kernel_3d(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int N,
    const int C,
    const int H,
    const int W,
    const float eps,
    const int H_blocks,
    const int W_blocks
) {
    // Map 3D grid to instance and spatial dimensions
    const int w_block = blockIdx.x;
    const int h_block = blockIdx.y;
    const int nc_idx = blockIdx.z;
    
    const int n = nc_idx / C;
    const int c = nc_idx % C;
    
    // Calculate spatial block dimensions
    const int w_size = (W + W_blocks - 1) / W_blocks;
    const int h_size = (H + H_blocks - 1) / H_blocks;
    
    const int w_start = w_block * w_size;
    const int h_start = h_block * h_size;
    const int w_end = min(w_start + w_size, W);
    const int h_end = min(h_start + h_size, H);

    // Instance offset
    const int instance_offset = (n * C + c) * H * W;
    const float* x_instance = x + instance_offset;
    float* y_instance = y + instance_offset;

    // Shared memory for partial sums
    __shared__ float s_mean, s_var;
    __shared__ float s_partial_sum[256];
    __shared__ float s_partial_sq_sum[256];

    // First pass: compute local sums
    float sum = 0.0f;
    float sq_sum = 0.0f;

    for (int h = h_start + threadIdx.y; h < h_end; h += blockDim.y) {
        for (int w = w_start + threadIdx.x; w < w_end; w += blockDim.x) {
            const float val = x_instance[h * W + w];
            sum += val;
            sq_sum += val * val;
        }
    }

    // Store partial results
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    s_partial_sum[tid] = sum;
    s_partial_sq_sum[tid] = sq_sum;
    __syncthreads();

    // Reduce within block
    if (tid == 0) {
        sum = 0.0f;
        sq_sum = 0.0f;
        for (int i = 0; i < blockDim.x * blockDim.y; ++i) {
            sum += s_partial_sum[i];
            sq_sum += s_partial_sq_sum[i];
        }
        
        // Atomic add to global memory for final reduction
        atomicAdd(&s_mean, sum);
        atomicAdd(&s_var, sq_sum);
    }
    __syncthreads();

    // Only one thread per block computes final statistics
    if (tid == 0 && h_block == 0 && w_block == 0) {
        const float HW = static_cast<float>(H * W);
        const float mean = s_mean / HW;
        const float variance = (s_var / HW) - (mean * mean);
        s_mean = mean;
        s_var = max(variance, 0.0f);
    }
    __syncthreads();

    // Second pass: normalize
    const float mean = s_mean;
    const float inv_std = rsqrtf(s_var + eps);
    const float w = weight ? weight[c] : 1.0f;
    const float b = bias ? bias[c] : 0.0f;

    for (int h = h_start + threadIdx.y; h < h_end; h += blockDim.y) {
        for (int w = w_start + threadIdx.x; w < w_end; w += blockDim.x) {
            const int idx = h * W + w;
            const float val = x_instance[idx];
            const float norm_val = (val - mean) * inv_std;
            y_instance[idx] = fmaf(norm_val, w, b);
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    if (weight.defined()) TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined()) TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input must be 4D");

    const int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    auto y = torch::empty_like(x);

    // Calculate grid dimensions
    const int H_blocks = (H + 31) / 32;
    const int W_blocks = (W + 31) / 32;
    
    dim3 threads(16, 16);  // 256 threads per block
    dim3 blocks(W_blocks, H_blocks, N * C);

    instance_norm_kernel_3d<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps),
        H_blocks, W_blocks
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CUDA) with 3D grid");
}