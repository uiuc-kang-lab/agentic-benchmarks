#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__global__ void batch_norm_ldg_kernel(
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
    const int num_rows = N * H;
    const int total_elements = num_rows * W;
    const int warp_size = 32;
    const int tid = threadIdx.x;
    const int lane = tid % warp_size;
    const int warp_id = tid / warp_size;
    const int num_warps = blockDim.x / warp_size;

    // Align accesses using 128-bit loads where possible
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    const int W4 = W / 4;
    const int W_remainder = W % 4;

    float local_sum = 0.f;
    float local_sum_sq = 0.f;

    // Process aligned 128-bit blocks
    for (int row = warp_id; row < num_rows; row += num_warps) {
        const int n = row / H;
        const int h = row % H;
        const int base = (n * C * H + c * H + h) * W;
        const int base4 = (base) / 4;

        for (int w4 = lane; w4 < W4; w4 += warp_size) {
            const float4 val4 = __ldg(&input4[base4 + w4]);
            const float* vals = reinterpret_cast<const float*>(&val4);
            for (int i = 0; i < 4; i++) {
                local_sum += vals[i];
                local_sum_sq += vals[i] * vals[i];
            }
        }

        // Process remaining elements
        if (lane >= W4 * 4 && lane < W) {
            const int idx = base + lane;
            const float val = __ldg(&input[idx]);
            local_sum += val;
            local_sum_sq += val * val;
        }
    }

    // Warp reduction
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    __shared__ float smem[64];
    if (lane == 0) {
        smem[warp_id] = local_sum;
        smem[warp_id + 32] = local_sum_sq;
    }
    __syncthreads();

    float mean = 0.f, var = 0.f;
    if (tid == 0) {
        float total_sum = 0.f, total_sum_sq = 0.f;
        for (int i = 0; i < num_warps; i++) {
            total_sum += smem[i];
            total_sum_sq += smem[i + 32];
        }
        mean = total_sum / total_elements;
        var = (total_sum_sq / total_elements) - (mean * mean);

        if (training) {
            running_mean[c] = (1.f - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.f - momentum) * running_var[c] + momentum * var;
        }
        smem[0] = mean;
        smem[1] = var;
    }
    __syncthreads();

    mean = smem[0];
    var = smem[1];
    const float inv_std = rsqrtf(var + eps);
    const float w_val = __ldg(&weight[c]);
    const float b_val = __ldg(&bias[c]);

    // Write aligned 128-bit blocks
    for (int row = warp_id; row < num_rows; row += num_warps) {
        const int n = row / H;
        const int h = row % H;
        const int base = (n * C * H + c * H + h) * W;
        const int base4 = base / 4;

        for (int w4 = lane; w4 < W4; w4 += warp_size) {
            float4 val4;
            float* vals = reinterpret_cast<float*>(&val4);
            const int idx = base4 + w4;
            const float4 in4 = __ldg(&input4[idx]);
            
            for (int i = 0; i < 4; i++) {
                vals[i] = (in4.x - mean) * inv_std * w_val + b_val;
            }
            output4[idx] = val4;
        }

        // Process remaining elements
        if (lane >= W4 * 4 && lane < W) {
            const int idx = base + lane;
            output[idx] = (__ldg(&input[idx]) - mean) * inv_std * w_val + b_val;
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
    size_t shared = 64 * sizeof(float);

    batch_norm_ldg_kernel<<<C, threads, shared>>>(
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
    m.def("forward", &forward_cuda, "LDG BatchNorm forward (CUDA)");
}