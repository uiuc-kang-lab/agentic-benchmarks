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
    const int row_id = blockIdx.y;
    const int num_rows = N * H;
    
    if (c >= C || row_id >= num_rows) return;

    const int warpSize = 32;
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int num_warps = blockDim.x / warpSize;

    const int n = row_id / H;
    const int h = row_id % H;
    const int base = n * C * H * W + c * H * W + h * W;

    extern __shared__ float smem[];
    float* warp_sums = smem;
    float* warp_sum_sq = smem + num_warps;

    float sum = 0.f, sum_sq = 0.f;
    const int W_aligned = W - (W % 4);

    // Vectorized accumulation
    for (int w = lane * 4; w < W_aligned; w += warpSize * 4) {
        float4 val = *reinterpret_cast<const float4*>(input + base + w);
        sum += val.x + val.y + val.z + val.w;
        sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    // Handle remaining elements
    for (int w = W_aligned + lane; w < W; w += warpSize) {
        float val = input[base + w];
        sum += val;
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    if (lane == 0) {
        warp_sums[warp_id] = sum;
        warp_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();

    float mean, var;
    if (tid == 0) {
        float total_sum = 0.f, total_sum_sq = 0.f;
        for (int i = 0; i < num_warps; ++i) {
            total_sum += warp_sums[i];
            total_sum_sq += warp_sum_sq[i];
        }
        
        if (training) {
            mean = total_sum / (num_rows * W);
            var = total_sum_sq / (num_rows * W) - mean * mean;
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
    const float w_val = weight[c];
    const float b_val = bias[c];

    // Vectorized write
    for (int w = lane * 4; w < W_aligned; w += warpSize * 4) {
        float4 val = *reinterpret_cast<const float4*>(input + base + w);
        float4 out;
        out.x = (val.x - mean) * inv_std * w_val + b_val;
        out.y = (val.y - mean) * inv_std * w_val + b_val;
        out.z = (val.z - mean) * inv_std * w_val + b_val;
        out.w = (val.w - mean) * inv_std * w_val + b_val;
        *reinterpret_cast<float4*>(output + base + w) = out;
    }

    // Handle remaining elements
    for (int w = W_aligned + lane; w < W; w += warpSize) {
        const int idx = base + w;
        output[idx] = (input[idx] - mean) * inv_std * w_val + b_val;
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
    const int num_warps = threads / 32;
    dim3 blocks(C, N * H);
    
    size_t shared_size = 2 * num_warps * sizeof(float);
    
    batch_norm_vectorized_kernel<<<blocks, threads, shared_size>>>(
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
