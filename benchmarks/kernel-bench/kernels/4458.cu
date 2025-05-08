#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template<int BLOCK_SIZE>
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int BLOCK_SIZE>
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    const int lane = threadIdx.x % warpSize;
    const int wid = threadIdx.x / warpSize;

    val = warpReduceSum<BLOCK_SIZE>(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE/warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum<BLOCK_SIZE>(val);
    
    return val;
}

template<int BLOCK_SIZE>
__global__ void instance_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int N,
    const int C,
    const int H,
    const int W,
    const float eps
) {
    const int instance_id = blockIdx.x;
    const int n = instance_id / C;
    const int c = instance_id % C;
    const int HW = H * W;
    
    constexpr int vector_size = 4; // Consider using a larger vector size if the hardware supports it for better memory coalescing.
    const int vector_elements = (HW + vector_size - 1) / vector_size;
    const int elements_per_thread = (vector_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    const float* x_instance = x + (n * C + c) * HW;
    float* y_instance = y + (n * C + c) * HW;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    #pragma unroll 2
    for (int i = 0; i < elements_per_thread; i++) {
        const int idx = (i * BLOCK_SIZE + threadIdx.x) * vector_size;
        const bool valid = idx < HW;
        
        float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (valid) {
            values = *reinterpret_cast<const float4*>(x_instance + idx);
        }
        
        sum += (valid ? values.x + values.y + values.z + values.w : 0.0f);
        sum_sq += (valid ? values.x * values.x + values.y * values.y + 
                          values.z * values.z + values.w * values.w : 0.0f);
    }
    
    const int remaining_start = vector_elements * vector_size;
    const int thread_remaining = threadIdx.x + remaining_start;
    if (thread_remaining < HW) {
        const float val = x_instance[thread_remaining];
        sum += val;
        sum_sq += val * val;
    }
    
    sum = blockReduceSum<BLOCK_SIZE>(sum);
    sum_sq = blockReduceSum<BLOCK_SIZE>(sum_sq);
    
    __shared__ float mean, inv_std;
    if (threadIdx.x == 0) {
        mean = sum / HW;
        const float var = fmaxf(sum_sq / HW - mean * mean, 0.0f);
        inv_std = rsqrtf(var + eps);
    }
    __syncthreads();
    
    const float w_val = weight ? weight[c] : 1.0f;
    const float b_val = bias ? bias[c] : 0.0f;
    
    #pragma unroll 2
    for (int i = 0; i < elements_per_thread; i++) {
        const int idx = (i * BLOCK_SIZE + threadIdx.x) * vector_size;
        const bool valid = idx < HW;
        
        if (valid) {
            float4 values = *reinterpret_cast<const float4*>(x_instance + idx);
            values.x = (values.x - mean) * inv_std * w_val + b_val;
            values.y = (values.y - mean) * inv_std * w_val + b_val;
            values.z = (values.z - mean) * inv_std * w_val + b_val;
            values.w = (values.w - mean) * inv_std * w_val + b_val;
            *reinterpret_cast<float4*>(y_instance + idx) = values;
        }
    }
    
    if (thread_remaining < HW) {
        y_instance[thread_remaining] = (x_instance[thread_remaining] - mean) * inv_std * w_val + b_val;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input must be 4D (N,C,H,W)");
    
    auto y = torch::empty_like(x);
    const int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    const int HW = H * W;
    const int blocks = N * C;
    
    if (HW <= 256) {
        instance_norm_kernel<64><<<blocks, 64>>>(
            x.data_ptr<float>(), y.data_ptr<float>(),
            weight.defined() ? weight.data_ptr<float>() : nullptr,
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            N, C, H, W, static_cast<float>(eps));
    } else if (HW <= 1024) {
        instance_norm_kernel<128><<<blocks, 128>>>(
            x.data_ptr<float>(), y.data_ptr<float>(),
            weight.defined() ? weight.data_ptr<float>() : nullptr,
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            N, C, H, W, static_cast<float>(eps));
    } else {
        instance_norm_kernel<256><<<blocks, 256>>>(
            x.data_ptr<float>(), y.data_ptr<float>(),
            weight.defined() ? weight.data_ptr<float>() : nullptr,
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            N, C, H, W, static_cast<float>(eps));
    }
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "InstanceNorm uniform flow (CUDA)");
}