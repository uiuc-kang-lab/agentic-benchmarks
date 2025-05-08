#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Define warp size
constexpr int WARP_SIZE = 32;

// Kernel that processes a subset of channels; 'channel_offset' indicates the starting channel index.
__global__ void batch_norm_pipeline_kernel(
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
    int W,
    int channel_offset) {

    int c = blockIdx.x + channel_offset;  // global channel index
    if (c >= C) return;

    int num_rows = N * H;            // Total rows per channel
    int total_elements = num_rows * W; // Total elements in this channel

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    float local_sum = 0.f;
    float local_sum_sq = 0.f;

    // Reduction: accumulate sum and sum of squares
    for (int row = warp_id; row < num_rows; row += num_warps) {
        int n = row / H;
        int h = row % H;
        int base = n * C * H * W + c * H * W + h * W;
        for (int w = lane; w < W; w += WARP_SIZE) {
            int idx = base + w;
            float val = input[idx];
            local_sum += val;
            local_sum_sq += val * val;
        }
    }

    // Warp-level reduction using shuffle
    unsigned mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum    += __shfl_down_sync(mask, local_sum, offset);
        local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
    }

    extern __shared__ float shared_mem[]; // Shared memory for inter-warp reduction
    float* warp_sums = shared_mem;
    float* warp_sum_sq_arr = &shared_mem[num_warps];

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
            running_var[c] = (1.f - momentum) * running_var[c] + momentum * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
        warp_sums[0] = mean;
        warp_sums[1] = var;
    }
    __syncthreads();

    mean = warp_sums[0];
    var  = warp_sums[1];
    float inv_std = rsqrtf(var + eps);
    
    float w_val = weight[c];
    float b_val = bias[c];

    // Normalization: apply batchnorm on each element
    for (int row = warp_id; row < num_rows; row += num_warps) {
        int n = row / H;
        int h = row % H;
        int base = n * C * H * W + c * H * W + h * W;
        for (int col = lane; col < W; col += WARP_SIZE) {
            int idx = base + col;
            float val = input[idx];
            output[idx] = (val - mean) * inv_std * w_val + b_val;
        }
    }
}

// Launcher that overlaps kernel execution with memory operations using CUDA streams
torch::Tensor forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {

    // Input validations
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

    const int threads = 256;  // threads per block, must be a multiple of warp size
    int num_warps = threads / WARP_SIZE;
    size_t shared_mem_bytes = 2 * num_warps * sizeof(float);

    // Partition the channels into groups to enable overlapping of work using multiple streams
    const int group_size = 16;  // adjustable parameter
    int num_groups = (C + group_size - 1) / group_size;
    std::vector<cudaStream_t> streams(num_groups);

    for (int i = 0; i < num_groups; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch a kernel for each group asynchronously
    for (int i = 0; i < num_groups; i++) {
        int channel_offset = i * group_size;
        int channels_in_group = std::min(group_size, C - channel_offset);
        // Launch kernel with grid dimension equal to the number of channels in this group
        batch_norm_pipeline_kernel<<<channels_in_group, threads, shared_mem_bytes, streams[i]>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            training,
            momentum,
            eps,
            output.data_ptr<float>(),
            N, C, H, W,
            channel_offset
        );
    }

    // If using Unified Memory, one could call cudaMemPrefetchAsync on input/output here to overlap
    // memory transfers with computation.

    // Synchronize and destroy streams
    for (int i = 0; i < num_groups; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Stream overlapped BatchNorm forward (CUDA)");
}
