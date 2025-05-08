#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Device function to compute partial sums and partial sum of squares
__device__ inline void compute_partial_sums(const float* __restrict__ input,
                                              int c, int N, int C, int H, int W,
                                              int tid, int stride,
                                              float &partial_sum, float &partial_sum_sq) {
    int num_elements = N * H * W;
    partial_sum = 0.0f;
    partial_sum_sq = 0.0f;
    for (int i = tid; i < num_elements; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        partial_sum += val;
        partial_sum_sq += val * val;
    }
}

// Device function for block-level reduction on shared memory array
__device__ inline void block_reduce(float* shared, int blockDim) {
    int tid = threadIdx.x;
    for (int s = blockDim / 2; s > 0; s >>= 1) {
        if(tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
}

// Device function to normalize a single value
__device__ inline float normalize_value(float val, float mean, float inv_std, float w, float b) {
    return (val - mean) * inv_std * w + b;
}

__global__ void batch_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    bool training,
    float momentum,
    float eps,
    float* __restrict__ output,
    int N, int C, int H, int W) {

    int c = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int num_elements = N * H * W;

    // Shared memory: first half for sum, second half for sum of squares
    extern __shared__ float smem[];
    float* sum_shared = smem;            // size: blockDim.x
    float* sum_sq_shared = &smem[blockDim.x]; // size: blockDim.x

    float mean, var;

    if (training) {
        float partial_sum, partial_sum_sq;
        compute_partial_sums(input, c, N, C, H, W, tid, stride, partial_sum, partial_sum_sq);
        sum_shared[tid] = partial_sum;
        sum_sq_shared[tid] = partial_sum_sq;
        __syncthreads();

        // Reduce the partial sums across the block
        block_reduce(sum_shared, blockDim.x);
        block_reduce(sum_sq_shared, blockDim.x);

        if (tid == 0) {
            float total_sum = sum_shared[0];
            float total_sum_sq = sum_sq_shared[0];
            mean = total_sum / num_elements;
            var = (total_sum_sq / num_elements) - (mean * mean);
            // Update running statistics
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
            // Store computed values in shared mem for Phase 2
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

    float inv_std = rsqrtf(var + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    // Phase 2: Normalize and write output
    for (int i = tid; i < num_elements; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w_idx = hw % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        float val = input[idx];
        output[idx] = normalize_value(val, mean, inv_std, w_val, b_val);
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

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty_like(input);

    int threads = 256;
    size_t shared_mem = 2 * threads * sizeof(float);

    batch_norm_kernel<<<C, threads, shared_mem>>>(
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
    m.def("forward", &forward_cuda, "BatchNorm forward (CUDA)");
}
