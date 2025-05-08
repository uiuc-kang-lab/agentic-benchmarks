#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__global__ void batch_norm_vectorized_kernel(
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

    float local_sum = 0.f;
    float local_sum_sq = 0.f;

    // Vectorized accumulation loop
    for (int row = warp_id; row < num_rows; row += num_warps) {
        const int n = row / H;
        const int h = row % H;
        const int base = n * C * H * W + c * H * W + h * W;

        int w = lane * 4;
        while (w < W) {
            if (w + 4 <= W) {
                const float4 vec = *reinterpret_cast<const float4*>(&input[base + w]);
                local_sum += vec.x + vec.y + vec.z + vec.w;
                local_sum_sq += vec.x*vec.x + vec.y*vec.y + vec.z*vec.z + vec.w*vec.w;
            } else {
                for (int i = 0; i < 4 && (w + i) < W; i++) {
                    const float val = input[base + w + i];
                    local_sum += val;
                    local_sum_sq += val * val;
                }
            }
            w += warpSize * 4;
        }
    }

    // Warp reduction
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    // Shared memory for warp results
    extern __shared__ float smem[];
    float* warp_sums = smem;
    float* warp_sum_sq = &smem[num_warps];

    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
        warp_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    // Final reduction
    float mean, var;
    if (tid == 0) {
        float total_sum = 0.f, total_sum_sq = 0.f;
        for (int i = 0; i < num_warps; i++) {
            total_sum += warp_sums[i];
            total_sum_sq += warp_sum_sq[i];
        }
        
        if (training) {
            mean = total_sum / total_elements;
            var = total_sum_sq / total_elements - mean * mean;
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
    var = warp_sums[1];
    const float inv_std = rsqrtf(var + eps);
    const float w_val = weight[c];
    const float b_val = bias[c];

    // Vectorized write loop
    for (int row = warp_id; row < num_rows; row += num_warps) {
        const int n = row / H;
        const int h = row % H;
        const int base = n * C * H * W + c * H * W + h * W;

        int w = lane * 4;
        while (w < W) {
            if (w + 4 <= W) {
                float4 val4;
                val4.x = (input[base + w] - mean) * inv_std * w_val + b_val;
                val4.y = (input[base + w + 1] - mean) * inv_std * w_val + b_val;
                val4.z = (input[base + w + 2] - mean) * inv_std * w_val + b_val;
                val4.w = (input[base + w + 3] - mean) * inv_std * w_val + b_val;
                *reinterpret_cast<float4*>(&output[base + w]) = val4;
            } else {
                for (int i = 0; i < 4 && (w + i) < W; i++) {
                    output[base + w + i] = (input[base + w + i] - mean) * inv_std * w_val + b_val;
                }
            }
            w += warpSize * 4;
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
    const size_t shared_size = 2 * (threads / 32) * sizeof(float);

    batch_norm_vectorized_kernel<<<C, threads, shared_size>>>(
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
    m.def("forward", &forward_cuda, "Vectorized BatchNorm forward (CUDA)");
}
