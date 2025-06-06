#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Constant memory for weight and bias
__constant__ float d_weight[1024];
__constant__ float d_bias[1024];

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

// Kernel with constant memory access
__global__ void instance_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
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
    const int items_per_thread = (HW + (blockDim.x * vector_size) - 1) / (blockDim.x * vector_size);
    const int aligned_HW = (HW / vector_size) * vector_size;

    for (int item = 0; item < items_per_thread; item++) {
        int idx = (item * blockDim.x + threadIdx.x) * vector_size;
        if (idx < aligned_HW) {
            float4 vec = __ldg(reinterpret_cast<const float4*>(x_ptr) + idx/vector_size);
            sum_val += vec.x + vec.y + vec.z + vec.w;
            sum_sq_val += vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
        }
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
    const float w = d_weight[c];
    const float b = d_bias[c];

    for (int item = 0; item < items_per_thread; item++) {
        int idx = (item * blockDim.x + threadIdx.x) * vector_size;
        if (idx < aligned_HW) {
            float4 vec = __ldg(reinterpret_cast<const float4*>(x_ptr) + idx/vector_size);
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

inline void cudaCheckError(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(err);
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input must be 4D (N,C,H,W)");

    auto y = torch::empty_like(x);
    const int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];

    // Copy weight and bias to constant memory
    if (weight.defined() && C <= 1024) {
        cudaCheckError(cudaMemcpyToSymbol(d_weight, weight.data_ptr<float>(), C * sizeof(float)));
    }
    if (bias.defined() && C <= 1024) {
        cudaCheckError(cudaMemcpyToSymbol(d_bias, bias.data_ptr<float>(), C * sizeof(float)));
    }
    
    const int threads = 128;
    const int blocks = N * C;

    instance_norm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "InstanceNorm with constant memory (CUDA)");
}