#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Helper device function: Warp-level reduction
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

// Helper device function: Block-level reduction using warp reductions
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

__global__ void instance_norm_kernel_optimized(
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
    const int n = instance_id / C;
    const int c = instance_id % C;
    const int HW = H * W;
    const int vector_size = 4;
    const int items_per_thread = (HW + (blockDim.x * vector_size) - 1) / (blockDim.x * vector_size);
    const int aligned_HW = (HW / vector_size) * vector_size;

    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;
    float sum_val = 0.0f, sum_sq_val = 0.0f;

    for (int item = 0; item < items_per_thread; ++item) {
        int idx = (item * blockDim.x + threadIdx.x) * vector_size;
        if (idx < aligned_HW) {
            float4 vec = __ldg(reinterpret_cast<const float4*>(x_ptr + idx));
            sum_val += (vec.x + vec.y + vec.z + vec.w);
            sum_sq_val += (vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w);
        }
        __syncthreads();
    }

    for (int i = aligned_HW + threadIdx.x; i < HW; i += blockDim.x) {
        float v = __ldg(x_ptr + i);
        sum_val += v;
        sum_sq_val += v * v;
    }

    sum_val = blockReduceSum(sum_val);
    sum_sq_val = blockReduceSum(sum_sq_val);

    __shared__ float sharedMean, sharedInvStd;
    if (threadIdx.x == 0) {
        float mean = sum_val / HW;
        float var = fmaxf(sum_sq_val / HW - mean * mean, 0.0f);
        sharedMean = mean;
        sharedInvStd = rsqrtf(var + eps);
    }
    __syncthreads();

    const float mean = sharedMean;
    const float inv_std = sharedInvStd;
    const float w = (weight != nullptr) ? __ldg(weight + c) : 1.0f;
    const float b = (bias != nullptr) ? __ldg(bias + c) : 0.0f;

    for (int item = 0; item < items_per_thread; ++item) {
        int idx = (item * blockDim.x + threadIdx.x) * vector_size;
        if (idx < aligned_HW) {
            float4 vec = __ldg(reinterpret_cast<const float4*>(x_ptr + idx));
            vec.x = (vec.x - mean) * inv_std * w + b;
            vec.y = (vec.y - mean) * inv_std * w + b;
            vec.z = (vec.z - mean) * inv_std * w + b;
            vec.w = (vec.w - mean) * inv_std * w + b;
            reinterpret_cast<float4*>(y_ptr)[idx/vector_size] = vec;
        }
    }

    for (int i = aligned_HW + threadIdx.x; i < HW; i += blockDim.x) {
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
    
    instance_norm_kernel_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        N, C, H, W, static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "InstanceNorm no warp divergence (CUDA)");
}
