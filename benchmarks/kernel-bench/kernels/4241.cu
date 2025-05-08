#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define STREAM_COUNT 4

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ void blockReduceSum(float &sum, float &sumSq) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    sum = warpReduceSum(sum);
    sumSq = warpReduceSum(sumSq);
    
    __shared__ float warpSum[32];
    __shared__ float warpSumSq[32];
    
    if (lane == 0) {
        warpSum[wid] = sum;
        warpSumSq[wid] = sumSq;
    }
    __syncthreads();
    
    if (wid == 0) {
        sum = (threadIdx.x < (blockDim.x / warpSize)) ? warpSum[lane] : 0;
        sumSq = (threadIdx.x < (blockDim.x / warpSize)) ? warpSumSq[lane] : 0;
        sum = warpReduceSum(sum);
        sumSq = warpReduceSum(sumSq);
    }
    __syncthreads();
}

__global__ void pipelined_batch_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    float* __restrict__ output,
    bool training,
    float momentum,
    float eps,
    int N, int C, int H, int W,
    int channels_per_stream) {
    
    const int c = blockIdx.x + blockIdx.y * channels_per_stream;
    if (c >= C) return;
    
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int num_elements = N * H * W;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int i = tid; i < num_elements; i += stride) {
        const int n = i / (H * W);
        const int hw = i % (H * W);
        const int h = hw / W;
        const int w = hw % W;
        const int idx = ((n * C + c) * H + h) * W + w;
        const float val = input[idx];
        sum += val;
        sum_sq += val * val;
    }
    
    blockReduceSum(sum, sum_sq);
    
    __shared__ float mean, inv_std;
    
    if (tid == 0) {
        mean = sum / num_elements;
        float var = (sum_sq / num_elements) - (mean * mean);
        if (training) {
            atomicExch(&running_mean[c], (1 - momentum) * running_mean[c] + momentum * mean);
            atomicExch(&running_var[c], (1 - momentum) * running_var[c] + momentum * var);
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
        inv_std = rsqrtf(var + eps);
    }
    __syncthreads();
    
    const float w = weight[c];
    const float b = bias[c];
    
    for (int i = tid; i < num_elements; i += stride) {
        const int n = i / (H * W);
        const int hw = i % (H * W);
        const int h = hw / W;
        const int w_idx = hw % W;
        const int idx = ((n * C + c) * H + h) * W + w_idx;
        const float val = input[idx];
        output[idx] = (val - mean) * inv_std * w + b;
    }
}

torch::Tensor pipelined_forward_cuda(
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
    
    std::vector<cudaStream_t> streams(STREAM_COUNT);
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int channels_per_stream = (C + STREAM_COUNT - 1) / STREAM_COUNT;
    const int threads = 256;
    
    for (int s = 0; s < STREAM_COUNT; s++) {
        const int stream_channels = std::min(channels_per_stream, C - s * channels_per_stream);
        if (stream_channels <= 0) break;
        
        dim3 grid(stream_channels, 1);
        
        pipelined_batch_norm_kernel<<<grid, threads, 0, streams[s]>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            output.data_ptr<float>(),
            training,
            momentum,
            eps,
            N, C, H, W,
            channels_per_stream
        );
    }
    
    // Synchronize all streams
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pipelined_forward_cuda, "Pipelined BatchNorm forward (CUDA)");
}