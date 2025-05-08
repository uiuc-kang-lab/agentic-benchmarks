#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Training kernel specialized to avoid divergent branches by separating training/inference paths
// and using warp shuffle reduction for uniform control flow
__global__ void batch_norm_kernel_train(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    float momentum,
    float eps,
    float* __restrict__ output,
    int N, int C, int H, int W) {

    // Each block handles one channel
    const int c = blockIdx.x;
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;

    // Use double precision for accumulation to improve numerical stability
    double sum = 0.0, sum_sq = 0.0;
    
    // Accumulate sums with improved memory coalescing
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        sum += static_cast<double>(val);
        sum_sq += static_cast<double>(val) * static_cast<double>(val);
    }

    // Perform warp-level reduction using shuffle intrinsics
    const unsigned int active_mask = __activemask();
    const unsigned int active_threads = __popc(active_mask);
    
    // Split double into high and low 32-bits for shuffle
    unsigned long long sum_bits = __double2ull_rd(sum);
    unsigned long long sum_sq_bits = __double2ull_rd(sum_sq);
    
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        unsigned int high_sum = __shfl_down_sync(active_mask, static_cast<unsigned int>(sum_bits >> 32), offset);
        unsigned int low_sum = __shfl_down_sync(active_mask, static_cast<unsigned int>(sum_bits), offset);
        unsigned int high_sum_sq = __shfl_down_sync(active_mask, static_cast<unsigned int>(sum_sq_bits >> 32), offset);
        unsigned int low_sum_sq = __shfl_down_sync(active_mask, static_cast<unsigned int>(sum_sq_bits), offset);
        
        if (threadIdx.x % warpSize < offset) {
            unsigned long long shuffled_sum = (static_cast<unsigned long long>(high_sum) << 32) | low_sum;
            unsigned long long shuffled_sum_sq = (static_cast<unsigned long long>(high_sum_sq) << 32) | low_sum_sq;
            sum += __ull2double_rd(shuffled_sum);
            sum_sq += __ull2double_rd(shuffled_sum_sq);
        }
    }

    int lane = threadIdx.x & (warpSize-1);
    int warpId = threadIdx.x >> 5;

    __shared__ double warp_sum[32];     // Using double for better precision
    __shared__ double warp_sum_sq[32];
    
    if (lane == 0) {
        warp_sum[warpId] = sum;
        warp_sum_sq[warpId] = sum_sq;
    }
    __syncthreads();

    // First thread in first warp reduces all warp results
    if (threadIdx.x == 0) {
        double total_sum = 0.0;
        double total_sum_sq = 0.0;
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        
        for (int i = 0; i < numWarps; ++i) {
            total_sum += warp_sum[i];
            total_sum_sq += warp_sum_sq[i];
        }

        float mean = static_cast<float>(total_sum / num_elements);
        float var = static_cast<float>(total_sum_sq / num_elements - (total_sum / num_elements) * (total_sum / num_elements));
        
        // Ensure variance is non-negative
        var = max(var, 0.0f);
        
        // Update running statistics
        float old_mean = running_mean[c];
        float old_var = running_var[c];
        running_mean[c] = (1.0f - momentum) * old_mean + momentum * mean;
        running_var[c] = (1.0f - momentum) * old_var + momentum * var;
        
        // Store for broadcasting
        warp_sum[0] = mean;
        warp_sum_sq[0] = var;
    }
    __syncthreads();
    float mean = warp_sum[0];
    float var = warp_sum_sq[0];
    float inv_std = rsqrtf(var + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    // Normalize and write output in a uniform loop
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w_idx = rem % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        float in_val = input[idx];
        output[idx] = (in_val - mean) * inv_std * w_val + b_val;
    }
}

// Inference kernel avoids any reduction and divergent branches by using precomputed running statistics
__global__ void batch_norm_kernel_infer(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    float* __restrict__ output,
    int N, int C, int H, int W) {

    const int c = blockIdx.x;
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;

    float mean = running_mean[c];
    float var = running_var[c];
    float inv_std = rsqrtf(var + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w_idx = rem % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        float in_val = input[idx];
        output[idx] = (in_val - mean) * inv_std * w_val + b_val;
    }
}

// Forward function chooses the appropriate kernel based on the training flag
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

    if (training) {
        batch_norm_kernel_train<<<C, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            momentum,
            eps,
            output.data_ptr<float>(),
            N, C, H, W
        );
    } else {
        batch_norm_kernel_infer<<<C, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            eps,
            output.data_ptr<float>(),
            N, C, H, W
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "BatchNorm forward (CUDA) with minimized warp divergence");
}
