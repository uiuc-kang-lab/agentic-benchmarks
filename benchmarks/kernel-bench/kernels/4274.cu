#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__global__ void batch_norm_kernel_unroll(
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
    
    const int c = blockIdx.x;
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    extern __shared__ float smem[];
    float* sum_shared = smem;
    float* sum_sq_shared = &smem[blockDim.x];

    float mean, var;
    
    if (training) {
        // Phase 1: Compute sum and sum of squares
        float sum = 0.0f, sum_sq = 0.0f;
        #pragma unroll
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

        sum_shared[tid] = sum;
        sum_sq_shared[tid] = sum_sq;
        __syncthreads();

        // Block reduction
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            if (tid < s) {
                sum_shared[tid] += sum_shared[tid + s];
                sum_sq_shared[tid] += sum_sq_shared[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float total_sum = sum_shared[0];
            const float total_sum_sq = sum_sq_shared[0];
            mean = total_sum / num_elements;
            var = (total_sum_sq / num_elements) - (mean * mean);
            
            // Update running statistics
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
            
            // Store in shared mem for next phase
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
    const float inv_std = rsqrtf(var + eps);
    const float w = weight[c];
    const float b = bias[c];

    #pragma unroll
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

torch::Tensor forward_cuda_unroll(
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
    
    const int threads = 256;
    const size_t shared_mem = 2 * threads * sizeof(float);
    
    batch_norm_kernel_unroll<<<C, threads, shared_mem>>>(
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
    m.def("forward_unroll", &forward_cuda_unroll, "BatchNorm forward with unrolling (CUDA)");
}