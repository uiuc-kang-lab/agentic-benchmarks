#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Optimized batch norm kernel that uses warp-level reduction combined with block-level synchronization
__global__ void batch_norm_kernel_optimized(
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

    // Each block handles one channel (c)
    const int c = blockIdx.x;
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;

    // Shared memory layout: use first part for warp partial sums and second part for partial sum of squares
    extern __shared__ float shared[];  // size: 2 * numWarps * sizeof(float)
    const int warpSize = 32;
    const int numWarps = blockSize / warpSize;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Each thread accumulates over its assigned elements
    for (int i = tid; i < num_elements; i += blockSize) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        sum += val;
        sum_sq += val * val;
    }

    // Intra-warp reduction using warp shuffle
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
        sum_sq += __shfl_down_sync(mask, sum_sq, offset);
    }

    // Write warp-level results to shared memory (one value per warp)
    if (lane == 0) {
        shared[warp_id] = sum;
        shared[warp_id + numWarps] = sum_sq;
    }
    __syncthreads();

    float mean, var;
    // Final reduction: thread 0 aggregates the partial sums
    if (tid == 0) {
        float final_sum = 0.0f;
        float final_sum_sq = 0.0f;
        for (int i = 0; i < numWarps; i++) {
            final_sum += shared[i];
            final_sum_sq += shared[i + numWarps];
        }
        mean = final_sum / num_elements;
        var = final_sum_sq / num_elements - mean * mean;
        if (training) {
            // Update running statistics
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
        } else {
            // In inference mode, use the provided running statistics
            mean = running_mean[c];
            var = running_var[c];
        }
        // Store computed mean and variance for broadcast
        shared[0] = mean;
        shared[1] = var;
    }
    __syncthreads();

    // Broadcast mean and variance to all threads
    mean = shared[0];
    var = shared[1];
    float inv_std = rsqrtf(var + eps);
    float scale = weight[c];
    float bias_val = bias[c];

    // Normalize and write output
    for (int i = tid; i < num_elements; i += blockSize) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w_idx = rem % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        float val = input[idx];
        output[idx] = (val - mean) * inv_std * scale + bias_val;
    }
}

// Host function to launch the optimized kernel
torch::Tensor forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {

    // Input validation
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
    const int warpSize = 32;
    int numWarps = threads / warpSize;
    // Allocate shared memory: 2 * numWarps floats
    const size_t shared_mem = 2 * numWarps * sizeof(float);

    // Launch one block per channel
    batch_norm_kernel_optimized<<<C, threads, shared_mem>>>(
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
    m.def("forward", &forward_cuda, "Optimized BatchNorm forward (CUDA)");
}
