#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Warp-level reduction for sum
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

// Block-level reduction using warp reduction
__inline__ __device__ float blockReduceSum(float val) {
    extern __shared__ float shared[];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < numWarps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// Modular device function to compute statistics (mean and inverse std) for an instance
__device__ void compute_stats(const float* x_ptr, float* shared, int HW, int tid, int blockSize, float eps, float &mean, float &inv_std) {
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    // Load elements into shared memory and accumulate partial sums
    for (int i = tid; i < HW; i += blockSize) {
        float val = x_ptr[i];
        shared[i] = val;
        local_sum += val;
        local_sum_sq += val * val;
    }
    local_sum = blockReduceSum(local_sum);
    local_sum_sq = blockReduceSum(local_sum_sq);

    __shared__ float s_mean;
    __shared__ float s_inv_std;

    if (tid == 0) {
        s_mean = local_sum / HW;
        float var = local_sum_sq / HW - s_mean * s_mean;
        if (var < 0.f) var = 0.f;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    mean = s_mean;
    inv_std = s_inv_std;
}

// Modular device function to apply normalization using precomputed statistics
__device__ void normalize_instance(const float* shared, float* y_ptr, int HW, int tid, int blockSize, float mean, float inv_std, float scale, float shift) {
    for (int i = tid; i < HW; i += blockSize) {
        float norm_val = (shared[i] - mean) * inv_std;
        y_ptr[i] = norm_val * scale + shift;
    }
}

// CUDA kernel for instance normalization using modular device functions
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
    // Each block processes one (N, C) instance
    int instance_id = blockIdx.x;
    int n = instance_id / C;
    int c = instance_id % C;
    int HW = H * W;

    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    extern __shared__ float shared_data[]; // shared memory for storing instance data
    float mean, inv_std;
    compute_stats(x_ptr, shared_data, HW, threadIdx.x, blockDim.x, eps, mean, inv_std);

    float scale = weight ? weight[c] : 1.0f;
    float shift = bias ? bias[c] : 0.0f;
    
    normalize_instance(shared_data, y_ptr, HW, threadIdx.x, blockDim.x, mean, inv_std, scale, shift);
}

// Forward function called from Python
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    if (weight.defined()) {
        TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    }
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    }

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D: (N, C, H, W)");
    
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];
    
    auto y = torch::empty_like(x);

    int threads = 256;
    int blocks = N * C;
    int shared_mem_size = H * W * sizeof(float);

    instance_norm_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps)
    );

    return y;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular Instance Normalization forward (CUDA)");
}
