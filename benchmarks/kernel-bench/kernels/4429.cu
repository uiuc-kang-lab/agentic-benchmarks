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
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

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
    const int instance_id = blockIdx.x;
    if (instance_id >= N * C) return;

    const int n = instance_id / C;
    const int c = instance_id % C;
    const int HW = H * W;
    const int vector_size = 4;

    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    float sum_val = 0.0f, sum_sq_val = 0.0f;

    int i = threadIdx.x * vector_size;
    const int aligned_HW = (HW / vector_size) * vector_size;

    while (i < aligned_HW) {
        float4 vec = __ldg(reinterpret_cast<const float4*>(x_ptr) + i/vector_size);
        float vals[4] = {vec.x, vec.y, vec.z, vec.w};
        
        #pragma unroll
        for (int j = 0; j < vector_size; ++j) {
            sum_val += vals[j];
            sum_sq_val += vals[j] * vals[j];
        }
        i += blockDim.x * vector_size;
    }

    for (i = aligned_HW + threadIdx.x; i < HW; i += blockDim.x) {
        float v = __ldg(x_ptr + i);
        sum_val += v;
        sum_sq_val += v * v;
    }

    sum_val = blockReduceSum(sum_val);
    sum_sq_val = blockReduceSum(sum_sq_val);

    __shared__ float sharedMean;
    __shared__ float sharedVar;
    if (threadIdx.x == 0) {
        float mean = sum_val / HW;
        float var = fmaxf(sum_sq_val / HW - mean * mean, 0.0f);
        sharedMean = mean;
        sharedVar = var;
    }
    __syncthreads();

    const float mean = sharedMean;
    const float inv_std = rsqrtf(sharedVar + eps);
    const float w = weight ? __ldg(weight + c) : 1.0f;
    const float b = bias ? __ldg(bias + c) : 0.0f;

    i = threadIdx.x * vector_size;
    while (i < aligned_HW) {
        float4 vec;
        vec.x = (__ldg(x_ptr + i) - mean) * inv_std * w + b;
        vec.y = (__ldg(x_ptr + i + 1) - mean) * inv_std * w + b;
        vec.z = (__ldg(x_ptr + i + 2) - mean) * inv_std * w + b;
        vec.w = (__ldg(x_ptr + i + 3) - mean) * inv_std * w + b;
        reinterpret_cast<float4*>(y_ptr)[i/vector_size] = vec;
        i += blockDim.x * vector_size;
    }

    for (i = aligned_HW + threadIdx.x; i < HW; i += blockDim.x) {
        y_ptr[i] = (__ldg(x_ptr + i) - mean) * inv_std * w + b;
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

    auto y = torch::empty_like(x);
    const int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    
    const int threads = 256;
    const int blocks = N * C;
    
    instance_norm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        N, C, H, W, static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "InstanceNorm optimized memory (CUDA)");
}
