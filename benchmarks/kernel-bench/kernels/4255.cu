#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define NUM_STREAMS 4

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void stream_compute_stats_kernel(
    const float* __restrict__ input,
    int N, int C, int H, int W,
    float* __restrict__ channel_mean,
    float* __restrict__ channel_var,
    int channels_per_stream,
    int stream_idx) {
    
    extern __shared__ float shared[];
    float* sum_shared = shared;
    float* sumsq_shared = &shared[blockDim.x];
    
    int c = blockIdx.x + stream_idx * channels_per_stream;
    if (c >= C) return;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int numElements = N * H * W;
    
    float sum = 0.0f, sumsq = 0.0f;
    for (int i = tid; i < numElements; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        sum += val;
        sumsq += val * val;
    }
    
    // Warp reduction
    sum = warpReduceSum(sum);
    sumsq = warpReduceSum(sumsq);
    
    if (tid < warpSize) {
        sum_shared[tid] = sum;
        sumsq_shared[tid] = sumsq;
    }
    __syncthreads();
    
    if (tid == 0) {
        float total_sum = 0.0f;
        float total_sumsq = 0.0f;
        for (int i = 0; i < warpSize; i++) {
            total_sum += sum_shared[i];
            total_sumsq += sumsq_shared[i];
        }
        float mean = total_sum / numElements;
        float var = (total_sumsq / numElements) - (mean * mean);
        channel_mean[c] = mean;
        channel_var[c] = var;
    }
}

__global__ void stream_normalize_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    const float* __restrict__ channel_mean,
    const float* __restrict__ channel_var,
    float* __restrict__ output,
    bool training,
    float momentum,
    float eps,
    int N, int C, int H, int W,
    int channels_per_stream,
    int stream_idx) {
    
    int c = blockIdx.x + stream_idx * channels_per_stream;
    if (c >= C) return;
    
    float mean = channel_mean[c];
    float var = channel_var[c];
    
    if (threadIdx.x == 0) {
        if (training) {
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
    }
    
    float invstd = rsqrtf(var + eps);
    float gamma = weight[c];
    float beta = bias[c];
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int numElements = N * H * W;
    
    for (int i = tid; i < numElements; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        output[idx] = (val - mean) * invstd * gamma + beta;
    }
}

torch::Tensor stream_forward_cuda(
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
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    auto output = torch::empty_like(input);
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto channel_mean = torch::empty({C}, options);
    auto channel_var = torch::empty({C}, options);
    
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    int channels_per_stream = (C + NUM_STREAMS - 1) / NUM_STREAMS;
    int threads = 512;
    size_t shared_mem = 2 * threads * sizeof(float);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        int stream_channels = std::min(channels_per_stream, C - i * channels_per_stream);
        if (stream_channels <= 0) break;
        
        dim3 grid_stats(stream_channels);
        stream_compute_stats_kernel<<<grid_stats, threads, shared_mem, streams[i]>>>(
            input.data_ptr<float>(),
            N, C, H, W,
            channel_mean.data_ptr<float>(),
            channel_var.data_ptr<float>(),
            channels_per_stream,
            i
        );
        
        dim3 grid_norm(stream_channels);
        stream_normalize_kernel<<<grid_norm, threads, 0, streams[i]>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            channel_mean.data_ptr<float>(),
            channel_var.data_ptr<float>(),
            output.data_ptr<float>(),
            training,
            momentum,
            eps,
            N, C, H, W,
            channels_per_stream,
            i
        );
    }
    
    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &stream_forward_cuda, "Stream Pipelined BatchNorm forward (CUDA)");
}