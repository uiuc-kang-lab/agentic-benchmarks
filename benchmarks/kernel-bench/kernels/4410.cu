#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized warp-level reduction using shuffle
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized block-level reduction using shared memory
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / warpSize)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// Optimized statistics computation using vectorized loads
__device__ void compute_instance_stats(
    const float* __restrict__ x_ptr,
    float* __restrict__ shared,
    int HW,
    int tid,
    int blockSize,
    float eps,
    float& mean,
    float& inv_std
) {
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Vectorized loading when possible
    if (HW % 4 == 0 && ((size_t)x_ptr % 16) == 0) {
        const float4* x_vec = reinterpret_cast<const float4*>(x_ptr);
        float4* shared_vec = reinterpret_cast<float4*>(shared);
        
        for (int i = tid; i < HW/4; i += blockSize) {
            float4 vals = x_vec[i];
            shared_vec[i] = vals;
            
            local_sum += vals.x + vals.y + vals.z + vals.w;
            local_sum_sq += vals.x * vals.x + vals.y * vals.y + 
                           vals.z * vals.z + vals.w * vals.w;
        }
    } else {
        for (int i = tid; i < HW; i += blockSize) {
            float val = x_ptr[i];
            shared[i] = val;
            local_sum += val;
            local_sum_sq += val * val;
        }
    }

    local_sum = blockReduceSum(local_sum);
    local_sum_sq = blockReduceSum(local_sum_sq);

    if (tid == 0) {
        mean = local_sum / HW;
        float var = fmaxf(local_sum_sq / HW - mean * mean, 0.0f);
        inv_std = rsqrtf(var + eps);
    }
    __syncthreads();
}

// Main kernel with adaptive block size and vectorized operations
__global__ void instance_norm_kernel_adaptive(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int H, int W,
    float eps
) {
    extern __shared__ float shared_data[];
    
    int c = blockIdx.x;
    int n = blockIdx.y;
    int HW = H * W;
    
    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    float mean, inv_std;
    compute_instance_stats(x_ptr, shared_data, HW, threadIdx.x, blockDim.x, eps, mean, inv_std);

    float scale = weight ? weight[c] : 1.0f;
    float shift = bias ? bias[c] : 0.0f;

    // Vectorized normalization when possible
    if (HW % 4 == 0 && ((size_t)y_ptr % 16) == 0) {
        float4* y_vec = reinterpret_cast<float4*>(y_ptr);
        const float4* shared_vec = reinterpret_cast<const float4*>(shared_data);
        
        for (int i = threadIdx.x; i < HW/4; i += blockDim.x) {
            float4 vals = shared_vec[i];
            float4 result;
            result.x = ((vals.x - mean) * inv_std) * scale + shift;
            result.y = ((vals.y - mean) * inv_std) * scale + shift;
            result.z = ((vals.z - mean) * inv_std) * scale + shift;
            result.w = ((vals.w - mean) * inv_std) * scale + shift;
            y_vec[i] = result;
        }
    } else {
        for (int i = threadIdx.x; i < HW; i += blockDim.x) {
            y_ptr[i] = ((shared_data[i] - mean) * inv_std) * scale + shift;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input must be 4D (N,C,H,W)");
    
    int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    auto y = torch::empty_like(x);
    
    int HW = H * W;
    // Adaptive block size selection based on instance size
    int block_size = (HW <= 256) ? 64 : 
                    (HW <= 1024) ? 128 :
                    (HW <= 4096) ? 256 : 512;
                    
    int blocks = N * C;
    int shared_mem = HW * sizeof(float);
    
    instance_norm_kernel_adaptive<<<blocks, block_size, shared_mem>>>(
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
    m.def("forward", &forward, "Adaptive Instance Normalization forward (CUDA)");
}