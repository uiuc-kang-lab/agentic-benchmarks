#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Kernel 1: Each block processes a portion of a single channel and computes partial sums.
// Grid dimensions: grid.x = number of channels, grid.y = blocksPerChannel
// Each block writes its partial sum and partial sum of squares into global arrays.
__global__ void batch_norm_reduce_kernel(
    const float* __restrict__ input,
    int N, int C, int H, int W,
    float* partial_sum,
    float* partial_sum_sq,
    int blocksPerChannel) {

    int c = blockIdx.x;           // channel index
    int blockId = blockIdx.y;       // block index within this channel
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int gridStride = blockSize * gridDim.y; // total threads per channel across blocks
    int numChannelElements = N * H * W;

    float sum = 0.0f, sumsq = 0.0f;
    // Each block processes a slice of the channel's elements
    for (int i = blockId * blockSize + tid; i < numChannelElements; i += gridStride) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        sum += val;
        sumsq += val * val;
    }

    // Shared memory reduction within the block
    extern __shared__ float sdata[]; // size: 2 * blockSize floats
    float* s_sum = sdata;
    float* s_sumsq = sdata + blockSize;
    s_sum[tid] = sum;
    s_sumsq[tid] = sumsq;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sumsq[tid] += s_sumsq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int globalIndex = c * blocksPerChannel + blockId;
        partial_sum[globalIndex] = s_sum[0];
        partial_sum_sq[globalIndex] = s_sumsq[0];
    }
}

// Kernel 2: Final reduction per channel. Each channel uses one block to reduce its partial sums
// and compute the final mean and variance. Also updates running stats if training is true.
__global__ void batch_norm_finalize_kernel(
    const float* partial_sum,
    const float* partial_sum_sq,
    int blocksPerChannel,
    int numChannelElements,
    bool training,
    float momentum,
    float* running_mean,
    float* running_var,
    float eps,
    float* mean_out,
    float* var_out) {

    int c = blockIdx.x; // one block per channel
    float sum = 0.0f, sumsq = 0.0f;

    // Each thread aggregates a portion of the partial sums
    for (int i = threadIdx.x; i < blocksPerChannel; i += blockDim.x) {
        int idx = c * blocksPerChannel + i;
        sum += partial_sum[idx];
        sumsq += partial_sum_sq[idx];
    }

    extern __shared__ float sdata[]; // shared mem: 2 * blockDim.x floats
    float* s_sum = sdata;
    float* s_sumsq = sdata + blockDim.x;
    s_sum[threadIdx.x] = sum;
    s_sumsq[threadIdx.x] = sumsq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sumsq[threadIdx.x] += s_sumsq[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean = s_sum[0] / numChannelElements;
        float var = s_sumsq[0] / numChannelElements - mean * mean;
        if (training) {
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
        mean_out[c] = mean;
        var_out[c] = var;
    }
}

// Kernel 3: Normalization kernel. Each block corresponds to a channel (grid.x) and blocks in y
// distribute the work evenly across the channel's elements.
__global__ void batch_norm_normalize_kernel(
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float eps) {

    int c = blockIdx.x; // channel index
    int numChannelElements = N * H * W;
    int tid = threadIdx.x + blockDim.x * blockIdx.y; // index within the channel
    int stride = blockDim.x * gridDim.y;
    
    float m = mean[c];
    float v = var[c];
    float invStd = rsqrtf(v + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    for (int i = tid; i < numChannelElements; i += stride) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        output[idx] = (val - m) * invStd * w_val + b_val;
    }
}

// Host function: Implements BatchNorm using three kernels to distribute workload evenly.
// 1. Reduction kernel: Computes partial sums per channel using multiple blocks per channel.
// 2. Finalize kernel: Aggregates partial sums to compute mean and variance and update running stats.
// 3. Normalization kernel: Applies normalization using computed statistics.

torch::Tensor forward_cuda(
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
    int numChannelElements = N * H * W;

    // Configure kernel launch parameters for reduction
    int threads = 256;
    int blocksPerChannel = (numChannelElements + threads - 1) / threads;
    
    // Allocate temporary buffers for partial reductions (size: C * blocksPerChannel)
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto partial_sum = torch::zeros({C * blocksPerChannel}, options);
    auto partial_sum_sq = torch::zeros({C * blocksPerChannel}, options);

    // Launch Kernel 1: Reduction kernel
    dim3 grid_reduce(C, blocksPerChannel);
    size_t shmem_reduce = 2 * threads * sizeof(float);
    batch_norm_reduce_kernel<<<grid_reduce, threads, shmem_reduce>>>(
        input.data_ptr<float>(),
        N, C, H, W,
        partial_sum.data_ptr<float>(),
        partial_sum_sq.data_ptr<float>(),
        blocksPerChannel
    );

    // Allocate tensors to store the final mean and variance per channel
    auto mean = torch::empty({C}, options);
    auto var = torch::empty({C}, options);

    // Launch Kernel 2: Final reduction kernel
    int threads_finalize = 256;
    dim3 grid_finalize(C);
    size_t shmem_finalize = 2 * threads_finalize * sizeof(float);
    batch_norm_finalize_kernel<<<grid_finalize, threads_finalize, shmem_finalize>>>(
        partial_sum.data_ptr<float>(),
        partial_sum_sq.data_ptr<float>(),
        blocksPerChannel,
        numChannelElements,
        training,
        momentum,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        mean.data_ptr<float>(),
        var.data_ptr<float>()
    );

    // Launch Kernel 3: Normalization kernel
    int blocksNorm = (numChannelElements + threads - 1) / threads;
    dim3 grid_norm(C, blocksNorm);
    auto output = torch::empty_like(input);
    batch_norm_normalize_kernel<<<grid_norm, threads>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        eps
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "BatchNorm forward with even workload distribution (CUDA)");
}
