#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Define a maximum number of channels that can be stored in constant memory
#ifndef MAX_CHANNELS
#define MAX_CHANNELS 4096
#endif

// Store frequently accessed, read-only data (weight and bias) in constant memory
__constant__ float d_weight[MAX_CHANNELS];
__constant__ float d_bias[MAX_CHANNELS];

// Device function: reduce values within a warp using shuffle instructions
__device__ void warp_reduce(float &sum, float &sum_sq) {
    const unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum    += __shfl_down_sync(mask, sum, offset);
        sum_sq += __shfl_down_sync(mask, sum_sq, offset);
    }
}

// Kernel: BatchNorm computation using constant memory for weight and bias
__global__ void batch_norm_constmem_kernel(
    const float* __restrict__ input,
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
    if (c >= C) return;

    int num_rows = N * H;              // number of rows per channel
    int total_elements = num_rows * W; // total elements in this channel

    const int warpSize = 32;
    int tid = threadIdx.x;
    int lane = tid % warpSize;         // lane index within a warp
    int warp_id = tid / warpSize;      // warp index within the block
    int num_warps = blockDim.x / warpSize;

    // Each thread accumulates partial sums
    float local_sum = 0.f;
    float local_sum_sq = 0.f;

    // Iterate over rows assigned to this warp
    for (int row = warp_id; row < num_rows; row += num_warps) {
        int n = row / H;       // image index
        int h = row % H;       // row index within image
        int base = n * C * H * W + c * H * W + h * W;
        // Process elements along the width dimension in a coalesced fashion
        for (int w = lane; w < W; w += warpSize) {
            int idx = base + w;
            float val = input[idx];
            local_sum += val;
            local_sum_sq += val * val;
        }
    }

    // Warp-level reduction
    warp_reduce(local_sum, local_sum_sq);

    // Use shared memory to accumulate sums across warps
    extern __shared__ float shmem[];  // Shared memory allocated dynamically
    float* warp_sums = shmem;          // first num_warps floats for sums
    float* warp_sum_sq_arr = &shmem[num_warps]; // next num_warps floats for sum of squares

    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
        warp_sum_sq_arr[warp_id] = local_sum_sq;
    }
    __syncthreads();

    float mean, var;
    if (tid == 0) {
        float total_sum = 0.f;
        float total_sum_sq = 0.f;
        for (int i = 0; i < num_warps; i++) {
            total_sum += warp_sums[i];
            total_sum_sq += warp_sum_sq_arr[i];
        }
        mean = total_sum / total_elements;
        var = total_sum_sq / total_elements - mean * mean;

        if (training) {
            running_mean[c] = (1.f - momentum) * running_mean[c] + momentum * mean;
            running_var[c]  = (1.f - momentum) * running_var[c]  + momentum * var;
        } else {
            mean = running_mean[c];
            var  = running_var[c];
        }
        // Store computed mean and variance in shared memory for broadcasting
        warp_sums[0] = mean;
        warp_sums[1] = var;
    }
    __syncthreads();

    // Broadcast the mean and variance to all threads
    mean = warp_sums[0];
    var  = warp_sums[1];
    float inv_std = rsqrtf(var + eps);

    // Access weight and bias from constant memory
    float w_val = d_weight[c];
    float b_val = d_bias[c];

    // Normalize input and write output using coalesced memory access
    for (int row = warp_id; row < num_rows; row += num_warps) {
        int n = row / H;
        int h = row % H;
        int base = n * C * H * W + c * H * W + h * W;
        for (int col = lane; col < W; col += warpSize) {
            int idx = base + col;
            float val = input[idx];
            output[idx] = (val - mean) * inv_std * w_val + b_val;
        }
    }
}

// Kernel launcher
torch::Tensor forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {

    // Input checks
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

    // Copy weight and bias to constant memory. Assumes C <= MAX_CHANNELS.
    cudaMemcpyToSymbol(d_weight, weight.data_ptr<float>(), C * sizeof(float));
    cudaMemcpyToSymbol(d_bias, bias.data_ptr<float>(),  C * sizeof(float));

    // Launch one block per channel
    const int threads = 256; // Should be a multiple of 32
    int num_warps = threads / 32;
    size_t shared_mem_bytes = 2 * num_warps * sizeof(float);

    batch_norm_constmem_kernel<<<C, threads, shared_mem_bytes>>>(
        input.data_ptr<float>(),
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
    m.def("forward", &forward_cuda, "BatchNorm forward with constant memory (CUDA)");
}
