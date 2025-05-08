#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define WARP_SIZE 32

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void hybrid_batchnorm_kernel(
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
    const int warpId = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int warpsPerBlock = blockDim.x / WARP_SIZE;
    const int numElements = N * H * W;

    extern __shared__ float smem[];
    float* warpSums = smem;
    float* warpSumSqs = &smem[warpsPerBlock];
    float* channelStats = &smem[2 * warpsPerBlock];

    float sum = 0.0f;
    float sumSq = 0.0f;

    const int vector_size = 4;
    float4 input_vec;
    
    #pragma unroll 2
    for (int i = tid; i < numElements/vector_size; i += blockDim.x) {
        const int base_idx = ((i * vector_size) / (H * W)) * (C * H * W) + 
                            c * H * W +
                            ((i * vector_size) % (H * W));
        
        input_vec = reinterpret_cast<const float4*>(input)[base_idx/vector_size];
        
        sum += input_vec.x + input_vec.y + input_vec.z + input_vec.w;
        sumSq += input_vec.x * input_vec.x + input_vec.y * input_vec.y + 
                 input_vec.z * input_vec.z + input_vec.w * input_vec.w;
    }

    const int remainStart = (numElements/vector_size) * vector_size;
    for (int i = remainStart + tid; i < numElements; i += blockDim.x) {
        const int n = i / (H * W);
        const int hw = i % (H * W);
        const int idx = n * C * H * W + c * H * W + hw;
        const float val = input[idx];
        sum += val;
        sumSq += val * val;
    }

    sum = warpReduceSum(sum);
    sumSq = warpReduceSum(sumSq);

    if (lane == 0) {
        warpSums[warpId] = sum;
        warpSumSqs[warpId] = sumSq;
    }

    __syncthreads();

    if (warpId == 0) {
        sum = (lane < warpsPerBlock) ? warpSums[lane] : 0.0f;
        sumSq = (lane < warpsPerBlock) ? warpSumSqs[lane] : 0.0f;
        
        sum = warpReduceSum(sum);
        sumSq = warpReduceSum(sumSq);

        if (lane == 0) {
            float mean = sum / numElements;
            float var = (sumSq / numElements) - (mean * mean);
            
            if (training) {
                atomicExch(&running_mean[c], (1 - momentum) * running_mean[c] + momentum * mean);
                atomicExch(&running_var[c], (1 - momentum) * running_var[c] + momentum * var);
            } else {
                mean = running_mean[c];
                var = running_var[c];
            }

            channelStats[0] = mean;
            channelStats[1] = rsqrtf(var + eps);
        }
    }

    __syncthreads();

    const float w = weight[c];
    const float b = bias[c];
    const float mean = channelStats[0];
    const float invStd = channelStats[1];

    #pragma unroll 2
    for (int i = tid; i < numElements/vector_size; i += blockDim.x) {
        const int base_idx = ((i * vector_size) / (H * W)) * (C * H * W) + 
                            c * H * W +
                            ((i * vector_size) % (H * W));
        
        input_vec = reinterpret_cast<const float4*>(input)[base_idx/vector_size];
        float4 output_vec;
        
        output_vec.x = (input_vec.x - mean) * invStd * w + b;
        output_vec.y = (input_vec.y - mean) * invStd * w + b;
        output_vec.z = (input_vec.z - mean) * invStd * w + b;
        output_vec.w = (input_vec.w - mean) * invStd * w + b;
        
        reinterpret_cast<float4*>(output)[base_idx/vector_size] = output_vec;
    }

    for (int i = remainStart + tid; i < numElements; i += blockDim.x) {
        const int n = i / (H * W);
        const int hw = i % (H * W);
        const int idx = n * C * H * W + c * H * W + hw;
        const float val = input[idx];
        output[idx] = (val - mean) * invStd * w + b;
    }
}