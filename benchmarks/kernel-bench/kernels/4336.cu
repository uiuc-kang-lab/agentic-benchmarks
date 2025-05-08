#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__global__ void batch_norm_warp_uniform_kernel(
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
    if (c >= C) return;

    const int warpSize = 32;
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int num_warps = blockDim.x / warpSize;
    const int num_rows = N * H;
    const int total_elements = num_rows * W;

    __shared__ float smem[2];
    float mean, var;

    if (training) {
        // Training path: compute statistics from input
        float local_sum = 0.f, local_sum_sq = 0.f;

        for (int row = warp_id; row < num_rows; row += num_warps) {
            int n = row / H;
            int h = row % H;
            int base = n * C * H * W + c * H * W + h * W;

            for (int w = lane; w < W; w += warpSize) {
                float val = input[base + w];
                local_sum += val;
                local_sum_sq += val * val;
            }
        }

        // Warp reduction
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
            local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
        }

        if (lane == 0) {
            atomicAdd(&smem[0], local_sum);
            atomicAdd(&smem[1], local_sum_sq);
        }
        __syncthreads();

        if (tid == 0) {
            mean = smem[0] / total_elements;
            var = (smem[1] / total_elements) - (mean * mean);
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
            smem[0] = mean;
            smem[1] = var;
        }
    } else {
        // Inference path: directly use running statistics
        if (tid == 0) {
            mean = running_mean[c];
            var = running_var[c];
            smem[0] = mean;
            smem[1] = var;
        }
    }
    __syncthreads();

    mean = smem[0];
    var = smem[1];
    const float inv_std = rsqrtf(var + eps);
    const float w_val = weight[c];
    const float b_val = bias[c];

    // Unified normalization with coalesced access
    for (int row = warp_id; row < num_rows; row += num_warps) {
        int n = row / H;
        int h = row % H;
        int base = n * C * H * W + c * H * W + h * W;

        for (int w = lane; w < W; w += warpSize) {
            float val = input[base + w];
            output[base + w] = (val - mean) * inv_std * w_val + b_val;
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

    CHECK_CUDA(input); CHECK_CONTIGUOUS(input);
    CHECK_CUDA(weight); CHECK_CONTIGUOUS(weight);
    CHECK_CUDA(bias); CHECK_CONTIGUOUS(bias);
    CHECK_CUDA(running_mean); CHECK_CONTIGUOUS(running_mean);
    CHECK_CUDA(running_var); CHECK_CONTIGUOUS(running_var);

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);
    const int threads = 256;
    const size_t shared_size = 2 * sizeof(float);

    batch_norm_warp_uniform_kernel<<<C, threads, shared_size>>>(
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
    m.def("forward", &forward_cuda, "Warp-uniform BatchNorm forward (CUDA)");
}
