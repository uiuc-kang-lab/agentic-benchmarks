#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__inline__ __device__ float warpReduceSum(float val) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
#else
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
#endif
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    __syncthreads();

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void instance_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N,
    int C,
    int H,
    int W,
    float eps
) {
    // Each block handles one instance (N,C pair)
    const int nc_idx = blockIdx.z;
    if (nc_idx >= N * C) return;

    const int n = nc_idx / C;
    const int c = nc_idx % C;
    
    // 2D thread block for spatial dimensions
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Calculate global indices
    const int x_idx = bx * blockDim.x + tx;
    const int y_idx = by * blockDim.y + ty;
    
    const int HW = H * W;
    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    // Shared memory for partial sums
    __shared__ float s_partial_sum[16][16];
    __shared__ float s_partial_square[16][16];

    // Initialize shared memory
    s_partial_sum[ty][tx] = 0.0f;
    s_partial_square[ty][tx] = 0.0f;

    // First pass: compute mean and variance
    float thread_sum = 0.0f;
    float thread_square = 0.0f;

    #pragma unroll 4
    for (int h = y_idx; h < H; h += gridDim.y * blockDim.y) {
        for (int w = x_idx; w < W; w += gridDim.x * blockDim.x) {
            const int idx = h * W + w;
            if (idx < HW) {
                float val = __ldg(x_ptr + idx);
                thread_sum += val;
                thread_square += val * val;
            }
        }
    }

    s_partial_sum[ty][tx] = thread_sum;
    s_partial_square[ty][tx] = thread_square;
    __syncthreads();

    // Reduce within thread block
    if (ty == 0) {
        for (int i = 1; i < blockDim.y; ++i) {
            s_partial_sum[0][tx] += s_partial_sum[i][tx];
            s_partial_square[0][tx] += s_partial_square[i][tx];
        }
    }
    __syncthreads();

    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;
    if (ty == 0) {
        sum_val = s_partial_sum[0][tx];
        sum_sq_val = s_partial_square[0][tx];
    }

    if (ty == 0) {
        sum_val = blockReduceSum(sum_val);
        sum_sq_val = blockReduceSum(sum_sq_val);
    }

    __shared__ float mean, inv_std;
    if (tx == 0 && ty == 0) {
        mean = sum_val / HW;
        float var = fmaxf(sum_sq_val / HW - mean * mean, 0.0f);
        inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    // Load weight and bias
    const float w = weight ? __ldg(weight + c) : 1.0f;
    const float b = bias ? __ldg(bias + c) : 0.0f;

    // Second pass: normalize
    #pragma unroll 4
    for (int h = y_idx; h < H; h += gridDim.y * blockDim.y) {
        for (int w = x_idx; w < W; w += gridDim.x * blockDim.x) {
            const int idx = h * W + w;
            if (idx < HW) {
                float val = __ldg(x_ptr + idx);
                y_ptr[idx] = (val - mean) * inv_std * w + b;
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input must be 4D (N,C,H,W)");

    const int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    auto y = torch::empty_like(x);

    // 2D thread blocks of 16x16
    dim3 threads(16, 16);
    
    // Grid dimensions
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y,
        N * C
    );

    instance_norm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "InstanceNorm forward (CUDA)");
}