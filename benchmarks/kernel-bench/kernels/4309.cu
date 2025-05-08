#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Kernel using warp-level reduction and shared memory atomics for safe accumulation
__global__ void batch_norm_kernel_atomic(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    bool training,
    float momentum,
    float eps,
    float* __restrict__ output,
    int N,
    int C,
    int H,
    int W) {

    // Each block processes one channel
    int c = blockIdx.x;
    int num_elements = N * H * W;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Shared memory layout:
    // sdata[0]: accumulator for sum
    // sdata[1]: accumulator for sum of squares
    // sdata[2]: final mean
    // sdata[3]: final variance
    extern __shared__ float sdata[];

    // Initialize shared accumulators (only one thread does this)
    if (tid == 0) {
        sdata[0] = 0.0f; // sum accumulator
        sdata[1] = 0.0f; // sum_sq accumulator
    }
    __syncthreads();

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Each thread accumulates over its assigned elements
    for (int i = tid; i < num_elements; i += blockSize) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Perform warp-level reduction using shuffle instructions
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
        local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
    }

    int lane = tid & 31; // tid % 32
    // Each warp leader atomically adds its reduced sum to shared memory
    if (lane == 0) {
        atomicAdd(&sdata[0], local_sum);
        atomicAdd(&sdata[1], local_sum_sq);
    }
    __syncthreads();

    float mean, var;
    if (tid == 0) {
        float sum_total = sdata[0];
        float sum_sq_total = sdata[1];
        mean = sum_total / num_elements;
        var = (sum_sq_total / num_elements) - (mean * mean);
        if (training) {
            // Update running statistics
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
        // Store computed mean and variance for broadcast
        sdata[2] = mean;
        sdata[3] = var;
    }
    __syncthreads();

    // Broadcast values to all threads
    mean = sdata[2];
    var = sdata[3];
    float inv_std = rsqrtf(var + eps); // Cached intermediate result for inv_std
    float gamma = weight[c];
    float beta = bias[c];

    // Phase 2: Normalize the input and write to output
    for (int i = tid; i < num_elements; i += blockSize) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        output[idx] = (val - mean) * inv_std * gamma + beta;
    }
}

// Host function to launch the kernel
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

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);

    const int threads = 256;
    // Shared memory for 4 floats: two for atomic accumulation and two for broadcasting statistics
    size_t shared_mem = 4 * sizeof(float);

    // Launch one block per channel
    batch_norm_kernel_atomic<<<C, threads, shared_mem>>>(
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
    m.def("forward", &forward_cuda, "Atomic Warp BatchNorm forward (CUDA)");
}
