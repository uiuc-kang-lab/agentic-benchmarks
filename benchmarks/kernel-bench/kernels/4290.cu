#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define ELEMENTS_PER_THREAD 4

__global__ void batch_norm_kernel_strided(
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
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int num_elements = N * H * W;
    const int elements_per_thread = (num_elements + stride - 1) / stride;

    extern __shared__ float smem[];
    float* sum_shared = smem;
    float* sum_sq_shared = &smem[blockDim.x];

    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;

    // Process multiple elements per thread in a strided fashion
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int idx_base = tid + i * stride;
        if (idx_base < num_elements) {
            const int n = idx_base / (H * W);
            const int hw = idx_base % (H * W);
            const int h = hw / W;
            const int w = hw % W;
            const int input_idx = ((n * C + c) * H + h) * W + w;
            const float val = input[input_idx];
            thread_sum += val;
            thread_sum_sq += val * val;
        }
    }

    sum_shared[tid] = thread_sum;
    sum_sq_shared[tid] = thread_sum_sq;
    __syncthreads();

    for (int s = stride/2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_shared[tid] += sum_shared[tid + s];
            sum_sq_shared[tid] += sum_sq_shared[tid + s];
        }
        __syncthreads();
    }

    float mean, var;
    if (tid == 0) {
        mean = sum_shared[0] / num_elements;
        var = (sum_sq_shared[0] / num_elements) - (mean * mean);
        
        if (training) {
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
        
        smem[0] = mean;
        smem[1] = var;
    }
    __syncthreads();

    mean = smem[0];
    var = smem[1];
    const float inv_std = rsqrtf(var + eps);
    const float w = weight[c];
    const float b = bias[c];

    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int idx_base = tid + i * stride;
        if (idx_base < num_elements) {
            const int n = idx_base / (H * W);
            const int hw = idx_base % (H * W);
            const int h = hw / W;
            const int w_idx = hw % W;
            const int output_idx = ((n * C + c) * H + h) * W + w_idx;
            const float val = input[output_idx];
            output[output_idx] = (val - mean) * inv_std * w + b;
        }
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
    
    batch_norm_kernel_strided<<<C, threads, shared_mem>>>(
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
    m.def("forward", &forward_cuda, "BatchNorm forward with strided processing (CUDA)");
}