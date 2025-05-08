#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template<int BLOCK_SIZE>
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

template<int BLOCK_SIZE>
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    val = warpReduceSum<BLOCK_SIZE>(val);
    if (lane == 0) {
        if (lane == 0) shared[lane] = val;
        val = warpReduceSum<BLOCK_SIZE>(val);
    }
    
    return val;
}

template<int BLOCK_SIZE>
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
    const int items_per_thread = (HW + (BLOCK_SIZE * vector_size) - 1) / (BLOCK_SIZE * vector_size);
    const int aligned_HW = (HW / vector_size) * vector_size;

    #pragma unroll 4
    for (int item = 0; item < items_per_thread; item++) {
        int idx = (item * BLOCK_SIZE + threadIdx.x) * vector_size;
        if (idx < aligned_HW) {
            float4 vec = __ldg(reinterpret_cast<const float4*>(x_ptr + idx));
            sum_val += vec.x + vec.y + vec.z + vec.w;
            sum_sq_val += vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
        }
    }

    for (int i = aligned_HW + threadIdx.x; i < HW; i += BLOCK_SIZE) {
        float v = __ldg(x_ptr + i);
        sum_val += v;
        sum_sq_val += v * v;
    }

    sum_val = blockReduceSum<BLOCK_SIZE>(sum_val);
    sum_sq_val = blockReduceSum<BLOCK_SIZE>(sum_sq_val);

    __shared__ float sharedMean, sharedInvStd, sharedWeight, sharedBias;
    if (threadIdx.x == 0) {
        float mean = sum_val / HW;
        float var = fmaxf(sum_sq_val / HW - mean * mean, 0.0f);
        sharedMean = mean;
        sharedInvStd = rsqrtf(var + eps);
        sharedWeight = weight ? __ldg(weight + c) : 1.0f;
        sharedBias = bias ? __ldg(bias + c) : 0.0f;
    }
    __syncthreads();

    const float mean = sharedMean;
    const float inv_std = sharedInvStd;
    const float w = sharedWeight;
    const float b = sharedBias;

    #pragma unroll 4
    for (int item = 0; item < items_per_thread; item++) {
        int idx = (item * BLOCK_SIZE + threadIdx.x) * vector_size;
        if (idx < aligned_HW) {
            float4 vec = __ldg(reinterpret_cast<const float4*>(x_ptr + idx));
            vec.x = (vec.x - mean) * inv_std * w + b;
            vec.y = (vec.y - mean) * inv_std * w + b;
            vec.z = (vec.z - mean) * inv_std * w + b;
            vec.w = (vec.w - mean) * inv_std * w + b;
            reinterpret_cast<float4*>(y_ptr)[idx/vector_size] = vec;
        }
    }

    for (int i = aligned_HW + threadIdx.x; i < HW; i += BLOCK_SIZE) {
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
    const int HW = H * W;
    const int blocks = N * C;
    
    if (HW <= 256) {
        instance_norm_kernel<64><<<blocks, 64>>>(x.data_ptr<float>(), y.data_ptr<float>(),
            weight.defined() ? weight.data_ptr<float>() : nullptr,
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            N, C, H, W, static_cast<float>(eps));
    } else if (HW <= 1024) {
        instance_norm_kernel<128><<<blocks, 128>>>(x.data_ptr<float>(), y.data_ptr<float>(),
            weight.defined() ? weight.data_ptr<float>() : nullptr,
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            N, C, H, W, static_cast<float>(eps));
    } else if (HW <= 4096) {
        instance_norm_kernel<256><<<blocks, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(),
            weight.defined() ? weight.data_ptr<float>() : nullptr,
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            N, C, H, W, static_cast<float>(eps));
    } else {
        instance_norm_kernel<512><<<blocks, 512>>>(x.data_ptr<float>(), y.data_ptr<float>(),
            weight.defined() ? weight.data_ptr<float>() : nullptr,
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            N, C, H, W, static_cast<float>(eps));
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "InstanceNorm with shared params optimization (CUDA)");
}
