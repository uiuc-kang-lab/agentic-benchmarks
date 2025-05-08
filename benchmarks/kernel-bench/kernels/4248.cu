#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define WARP_SIZE 32

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ void warpBroadcast(float &mean, float &var, int srcLane) {
    mean = __shfl_sync(0xffffffff, mean, srcLane);
    var = __shfl_sync(0xffffffff, var, srcLane);
}

__global__ void warp_optimized_batch_norm_kernel(
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

    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < numElements; i += blockDim.x) {
        const int n = i / (H * W);
        const int hw = i % (H * W);
        const int idx = ((n * C + c) * H + hw/W) * W + hw%W;
        const float val = input[idx];
        sum += val;
        sum_sq += val * val;
    }

    sum = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);

    if (lane == 0) {
        #pragma unroll
        for (int offset = warps_per_block/2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset * WARP_SIZE);
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset * WARP_SIZE);
        }
    }

    float mean, var;
    if (tid == 0) {
        mean = sum / numElements;
        var = (sum_sq / numElements) - (mean * mean);
        
        if (training) {
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int w = 1; w < warps_per_block; w++) {
            __shfl_sync(0xffffffff, mean, 0, WARP_SIZE);
            __shfl_sync(0xffffffff, var, 0, WARP_SIZE);
        }
    }
    warpBroadcast(mean, var, 0);

    const float inv_std = rsqrtf(var + eps);
    const float w = weight[c];
    const float b = bias[c];

    #pragma unroll 4
    for (int i = tid; i < numElements; i += blockDim.x) {
        const int n = i / (H * W);
        const int hw = i % (H * W);
        const int idx = ((n * C + c) * H + hw/W) * W + hw%W;
        const float val = input[idx];
        output[idx] = (val - mean) * inv_std * w + b;
    }
}

torch::Tensor warp_optimized_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {
    
    CHECK_CUDA(input); CHECK_CUDA(weight); CHECK_CUDA(bias);
    CHECK_CUDA(running_mean); CHECK_CUDA(running_var);
    CHECK_CONTIGUOUS(input); CHECK_CONTIGUOUS(weight); CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(running_mean); CHECK_CONTIGUOUS(running_var);

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);
    
    const int threads = 256;
    warp_optimized_batch_norm_kernel<<<C, threads>>>(
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
    m.def("forward", &warp_optimized_forward_cuda, "Warp-optimized BatchNorm forward (CUDA)");
}