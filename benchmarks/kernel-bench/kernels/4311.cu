#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Define a maximum number of channels that can be stored in constant memory
// Adjust MAX_CHANNELS as needed based on expected tensor sizes and hardware limits
#define MAX_CHANNELS 4096

// Declare constant memory for weight and bias; these are read-only in the kernel
__constant__ float d_weight[MAX_CHANNELS];
__constant__ float d_bias[MAX_CHANNELS];

__global__ void batch_norm_kernel(
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
    
    const int c = blockIdx.x;
    if (c >= C) return;
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    extern __shared__ float smem[];
    // First half for reduction of sum, second half for sum of squares
    float* sum_shared = smem;
    float* sum_sq_shared = &smem[blockDim.x];

    float mean, var;
    
    if (training) {
        // Phase 1: Compute sum and sum of squares for the current channel
        float sum = 0.0f, sum_sq = 0.0f;
        for (int i = tid; i < num_elements; i += stride) {
            int n = i / (H * W);
            int hw = i % (H * W);
            int h = hw / W;
            int w = hw % W;
            int idx = ((n * C + c) * H + h) * W + w;
            float val = input[idx];
            sum += val;
            sum_sq += val * val;
        }

        sum_shared[tid] = sum;
        sum_sq_shared[tid] = sum_sq;
        __syncthreads();
        
        // Block reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sum_shared[tid] += sum_shared[tid + s];
                sum_sq_shared[tid] += sum_sq_shared[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            float total_sum = sum_shared[0];
            float total_sum_sq = sum_sq_shared[0];
            mean = total_sum / num_elements;
            var = (total_sum_sq / num_elements) - (mean * mean);
            
            // Update running statistics
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
            
            // Save mean and variance in shared memory for reuse in Phase 2
            smem[0] = mean;
            smem[1] = var;
        }
        __syncthreads();
        
        mean = smem[0];
        var = smem[1];
    } else {
        mean = running_mean[c];
        var = running_var[c];
    }

    // Phase 2: Normalize and write output
    float inv_std = rsqrtf(var + eps);
    // Access weight and bias from constant memory
    float w = d_weight[c];
    float b = d_bias[c];

    for (int i = tid; i < num_elements; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w_idx = hw % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        float val = input[idx];
        output[idx] = (val - mean) * inv_std * w + b;
    }
}

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
    
    TORCH_CHECK(C <= MAX_CHANNELS, "Channel count (" + std::to_string(C) + ") exceeds the constant memory capacity (" + std::to_string(MAX_CHANNELS) + ")");
    
    // Copy weight and bias from device memory into constant memory
    cudaMemcpyToSymbol(d_weight, weight.data_ptr<float>(), C * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(d_bias, bias.data_ptr<float>(), C * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const size_t shared_mem = 2 * threads * sizeof(float);
    
    // Each block handles one channel
    batch_norm_kernel<<<C, threads, shared_mem>>>(
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
