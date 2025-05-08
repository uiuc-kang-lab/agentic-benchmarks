#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define WARP_SIZE 32

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void optimized_batch_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    bool training,
    float momentum,
    float eps,
    float* __restrict__ output,
    int N, int C, int H, int W) {

    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int numElements = N * H * W;

    // Shared memory for partial sums
    extern __shared__ float shared[];
    float* sPartialSum = shared;
    float* sPartialSqSum = &shared[warps_per_block];

    // Initialize accumulators
    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Vector loading - process 4 elements at a time when possible
    const int vec_size = 4;
    const int vec_elements = numElements / vec_size;
    const int base_idx = c * N * H * W;
    
    // Vectorized loading and accumulation
    #pragma unroll 4
    for (int i = tid; i < vec_elements; i += blockDim.x) {
        const float4* vec_input = reinterpret_cast<const float4*>(&input[base_idx + i * vec_size]);
        float4 values = *vec_input;
        
        sum += values.x + values.y + values.z + values.w;
        sum_sq += values.x * values.x + values.y * values.y + 
                  values.z * values.z + values.w * values.w;
    }

    // Handle remaining elements
    for (int i = vec_elements * vec_size + tid; i < numElements; i += blockDim.x) {
        const float val = input[base_idx + i];
        sum += val;
        sum_sq += val * val;
    }

    // Warp-level reduction
    sum = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);

    // First thread in each warp writes to shared memory
    if (lane == 0) {
        sPartialSum[wid] = sum;
        sPartialSqSum[wid] = sum_sq;
    }
    __syncthreads();

    // Final reduction using first warp
    if (wid == 0) {
        sum = (lane < warps_per_block) ? sPartialSum[lane] : 0.0f;
        sum_sq = (lane < warps_per_block) ? sPartialSqSum[lane] : 0.0f;

        // Final warp reduction
        sum = warpReduceSum(sum);
        sum_sq = warpReduceSum(sum_sq);

        // First thread computes final statistics
        if (lane == 0) {
            float mean = sum / numElements;
            float variance = (sum_sq / numElements) - (mean * mean);

            if (training) {
                running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
                running_var[c] = (1.0f - momentum) * running_var[c] + momentum * variance;
            } else {
                mean = running_mean[c];
                variance = running_var[c];
            }

            // Store in shared memory for broadcasting
            sPartialSum[0] = mean;
            sPartialSqSum[0] = variance;
        }
    }
    __syncthreads();

    // Broadcast mean and variance to all threads
    const float mean = sPartialSum[0];
    const float variance = sPartialSqSum[0];
    const float inv_std = rsqrtf(variance + eps);
    const float w = weight[c];
    const float b = bias[c];

    // Normalize with vectorized operations where possible
    #pragma unroll 4
    for (int i = tid; i < vec_elements; i += blockDim.x) {
        float4* vec_output = reinterpret_cast<float4*>(&output[base_idx + i * vec_size]);
        float4 values = reinterpret_cast<const float4*>(&input[base_idx + i * vec_size])[0];
        
        values.x = (values.x - mean) * inv_std * w + b;
        values.y = (values.y - mean) * inv_std * w + b;
        values.z = (values.z - mean) * inv_std * w + b;
        values.w = (values.w - mean) * inv_std * w + b;
        
        *vec_output = values;
    }

    // Handle remaining elements
    for (int i = vec_elements * vec_size + tid; i < numElements; i += blockDim.x) {
        const float val = input[base_idx + i];
        output[base_idx + i] = (val - mean) * inv_std * w + b;
    }
}

torch::Tensor optimized_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {

    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    CHECK_CUDA(running_mean);
    CHECK_CUDA(running_var);

    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(running_mean);
    CHECK_CONTIGUOUS(running_var);

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);

    const int threads = 512;
    const int warps_per_block = threads / WARP_SIZE;
    const size_t shared_mem_size = 2 * warps_per_block * sizeof(float);

    optimized_batch_norm_kernel<<<C, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        training,
        momentum,
        eps,
        output.data_ptr<float>(),
        N, C, H, W
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_forward_cuda, "Optimized BatchNorm forward (CUDA)");
}