#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Declare constant memory for weight and bias (assuming maximum channels of 4096)
__constant__ float c_weight[4096];
__constant__ float c_bias[4096];

// Warp-level reduction
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

// Block-level reduction using warp reductions
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // one warp per 32 threads
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    
    val = warpReduceSum(val);
    __syncwarp();

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncwarp();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// CUDA kernel for Instance Normalization using constant memory for weight and bias
// The kernel performs vectorized loads and stores using float4 for efficiency.
__global__ void instance_norm_kernel_const(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N,
    int C,
    int H,
    int W,
    float eps
) {
    // Each block handles one (n, c) instance
    int instance_id = blockIdx.x;
    if (instance_id >= N * C) return;

    int n = instance_id / C;
    int c = instance_id % C;
    int HW = H * W;

    // Pointers to the start of the current instance for input and output
    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    // Vectorization parameters
    const int vecSize = 4;
    int aligned_HW = (HW / vecSize) * vecSize;

    // First pass: compute sum and sum of squares using vectorized loads
    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;
    int index = threadIdx.x * vecSize;
    int stride = blockDim.x * vecSize;

    while (index < aligned_HW) {
        float4 data = reinterpret_cast<const float4*>(x_ptr)[index / vecSize];
        sum_val   += data.x + data.y + data.z + data.w;
        sum_sq_val += data.x * data.x + data.y * data.y + data.z * data.z + data.w * data.w;
        index += stride;
    }
    
    // Process remaining elements (if HW is not divisible by 4)
    for (int i = aligned_HW + threadIdx.x; i < HW; i += blockDim.x) {
        float val = x_ptr[i];
        sum_val += val;
        sum_sq_val += val * val;
    }

    // Reduce sums to obtain global sum and sum of squares for this instance
    sum_val = blockReduceSum(sum_val);
    sum_sq_val = blockReduceSum(sum_sq_val);

    __shared__ float s_mean;
    __shared__ float s_var;
    if (threadIdx.x == 0) {
        float mean = sum_val / HW;
        float var = sum_sq_val / HW - mean * mean;
        s_mean = mean;
        s_var = (var < 0.f ? 0.f : var);
    }
    __syncwarp();

    float mean = s_mean;
    float inv_std = rsqrtf(s_var + eps);

    // Load scaling (gamma) and shifting (beta) factors from constant memory
    float gamma = c_weight[c];
    float beta = c_bias[c];

    // Second pass: apply normalization and optional scaling/shift using vectorized stores
    index = threadIdx.x * vecSize;
    while (index < aligned_HW) {
        float4 data = reinterpret_cast<const float4*>(x_ptr)[index / vecSize];
        float4 out;
        out.x = (data.x - mean) * inv_std * gamma + beta;
        out.y = (data.y - mean) * inv_std * gamma + beta;
        out.z = (data.z - mean) * inv_std * gamma + beta;
        out.w = (data.w - mean) * inv_std * gamma + beta;
        reinterpret_cast<float4*>(y_ptr)[index / vecSize] = out;
        index += stride;
    }
    
    // Process any leftover elements
    for (int i = aligned_HW + threadIdx.x; i < HW; i += blockDim.x) {
        float val = x_ptr[i];
        y_ptr[i] = (val - mean) * inv_std * gamma + beta;
    }
}

// Forward function called from Python
// Copies weight and bias data to constant memory and launches the kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a CUDA tensor");
    TORCH_CHECK(weight.defined(), "Weight tensor must be defined");
    TORCH_CHECK(bias.defined(), "Bias tensor must be defined");

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D: (N, C, H, W)");

    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    // Ensure constant memory can hold the weight and bias arrays
    TORCH_CHECK(C <= 4096, "Number of channels exceeds constant memory limit");

    // Copy weight and bias to constant memory (using device-to-device copy)
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), C * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_bias, bias.data_ptr<float>(), C * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    auto y = torch::empty_like(x);
    int threads = 256;
    int blocks = N * C;
    
    instance_norm_kernel_const<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward with constant memory (CUDA)");
}
