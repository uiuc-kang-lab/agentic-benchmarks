#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

typedef float4 vec4;

__device__ void vectorized_compute_sums(const float* input, int base, int W, int lane, float& sum, float& sum_sq) {
    const int vec_size = 4;
    int vec_lane = lane;
    int vec_count = W / vec_size;
    int remainder = W % vec_size;

    // Process vectorized elements
    for (int w = vec_lane; w < vec_count; w += 32) {
        int idx = base + w * vec_size;
        vec4 val = reinterpret_cast<const vec4*>(input + idx)[0];
        sum += val.x + val.y + val.z + val.w;
        sum_sq += val.x*val.x + val.y*val.y + val.z*val.z + val.w*val.w;
    }

    // Process remaining elements
    int scalar_base = base + vec_count * vec_size;
    if (vec_lane < remainder) {
        float val = input[scalar_base + vec_lane];
        sum += val;
        sum_sq += val * val;
    }
}

__device__ void vectorized_normalize(float* output, const float* input, int base, int W, int lane, float mean, float inv_std, float w, float b) {
    const int vec_size = 4;
    int vec_lane = lane;
    int vec_count = W / vec_size;
    int remainder = W % vec_size;

    // Process vectorized elements
    for (int w = vec_lane; w < vec_count; w += 32) {
        int idx = base + w * vec_size;
        vec4 val_in = reinterpret_cast<const vec4*>(input + idx)[0];
        vec4 val_out;
        
        val_out.x = (val_in.x - mean) * inv_std * w + b;
        val_out.y = (val_in.y - mean) * inv_std * w + b;
        val_out.z = (val_in.z - mean) * inv_std * w + b;
        val_out.w = (val_in.w - mean) * inv_std * w + b;
        
        reinterpret_cast<vec4*>(output + idx)[0] = val_out;
    }

    // Process remaining elements
    int scalar_base = base + vec_count * vec_size;
    if (vec_lane < remainder) {
        int idx = scalar_base + vec_lane;
        output[idx] = (input[idx] - mean) * inv_std * w + b;
    }
}

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

    int c = blockIdx.x;
    if (c >= C) return;

    const int warp_size = 32;
    int tid = threadIdx.x;
    int lane = tid % warp_size;
    int warp_id = tid / warp_size;
    int num_warps = blockDim.x / warp_size;

    float sum = 0.0f, sum_sq = 0.0f;
    int num_rows = N * H;

    // Phase 1: Vectorized sum reduction
    for (int row = warp_id; row < num_rows; row += num_warps) {
        int n = row / H;
        int h = row % H;
        int base = n * C * H * W + c * H * W + h * W;
        vectorized_compute_sums(input, base, W, lane, sum, sum_sq);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    extern __shared__ float smem[];
    float* warp_sums = smem;
    float* warp_sqsums = &smem[num_warps];

    if (lane == 0) {
        warp_sums[warp_id] = sum;
        warp_sqsums[warp_id] = sum_sq;
    }
    __syncthreads();

    // Final reduction
    float mean, var;
    if (tid == 0) {
        float total_sum = 0, total_sqsum = 0;
        for (int i = 0; i < num_warps; ++i) {
            total_sum += warp_sums[i];
            total_sqsum += warp_sqsums[i];
        }
        mean = total_sum / (num_rows * W);
        var = total_sqsum / (num_rows * W) - mean * mean;

        if (training) {
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
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
    float inv_std = rsqrtf(var + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    // Phase 2: Vectorized normalization
    for (int row = warp_id; row < num_rows; row += num_warps) {
        int n = row / H;
        int h = row % H;
        int base = n * C * H * W + c * H * W + h * W;
        vectorized_normalize(output, input, base, W, lane, mean, inv_std, w_val, b_val);
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

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty_like(input);

    const int threads = 256;
    int num_warps = threads / 32;
    size_t shared_size = 2 * num_warps * sizeof(float);

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
