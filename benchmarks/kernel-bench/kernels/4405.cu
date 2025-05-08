#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
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

    val = (threadIdx.x < (blockDim.x/warpSize)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void instance_norm_kernel(
    const float4* __restrict__ x4,
    float4* __restrict__ y4,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int N, const int C, const int HW4,
    const float eps
) {
    const int idx = blockIdx.x;
    const int n = idx / C;
    const int c = idx % C;
    
    extern __shared__ float4 shared[];
    const float4* x_ptr = x4 + (n * C + c) * HW4;
    float4* y_ptr = y4 + (n * C + c) * HW4;
    
    float sum = 0.0f, sum_sq = 0.0f;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < HW4; i += blockDim.x) {
        float4 val4 = x_ptr[i];
        shared[i] = val4;
        
        sum += val4.x + val4.y + val4.z + val4.w;
        sum_sq += val4.x*val4.x + val4.y*val4.y + val4.z*val4.z + val4.w*val4.w;
    }
    
    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);
    
    __shared__ float mean, invstd;
    if (threadIdx.x == 0) {
        mean = sum / (HW4 * 4);
        float var = fmaxf(sum_sq/(HW4 * 4) - mean*mean, 0.0f);
        invstd = rsqrtf(var + eps);
    }
    __syncthreads();
    
    const float scale = weight ? weight[c] : 1.0f;
    const float shift = bias ? bias[c] : 0.0f;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < HW4; i += blockDim.x) {
        float4 val4 = shared[i];
        val4.x = (val4.x - mean) * invstd * scale + shift;
        val4.y = (val4.y - mean) * invstd * scale + shift;
        val4.z = (val4.z - mean) * invstd * scale + shift;
        val4.w = (val4.w - mean) * invstd * scale + shift;
        y_ptr[i] = val4;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be NCHW");
    TORCH_CHECK(x.size(2) * x.size(3) % 4 == 0, "HW must be multiple of 4");
    
    const int N = x.size(0), C = x.size(1);
    const int HW4 = (x.size(2) * x.size(3)) / 4;
    
    auto y = torch::empty_like(x);
    
    const int block_size = std::min(256, HW4);
    const dim3 grid(N * C);
    const int shared_mem = HW4 * sizeof(float4);
    
    instance_norm_kernel<<<grid, block_size, shared_mem>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(y.data_ptr<float>()),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        N, C, HW4,
        static_cast<float>(eps)
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Norm forward (CUDA)");
}