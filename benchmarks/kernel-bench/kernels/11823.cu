#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline float compute_kldiv_value(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ inline float block_reduce_sum(float val, int tid, int block_size) {
    __shared__ float warp_sums[32];
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    val = warp_reduce_sum(val);

    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < (block_size + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }

    __syncthreads();
    return val;
}

__global__ void kl_div_kernel_combined(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    int vec_count = n / 4;

    for (int i = gid; i < vec_count; i += stride) {
        int base = i * 4;
        float4 log_vec = reinterpret_cast<const float4*>(log_predictions)[i];
        float4 target_vec = reinterpret_cast<const float4*>(targets)[i];

        local_sum += compute_kldiv_value(log_vec.x, target_vec.x)
                   + compute_kldiv_value(log_vec.y, target_vec.y)
                   + compute_kldiv_value(log_vec.z, target_vec.z)
                   + compute_kldiv_value(log_vec.w, target_vec.w);
    }

    float block_sum = block_reduce_sum(local_sum, tid, blockDim.x);

    if (tid == 0) {
        atomicAdd(output, block_sum);
    }

    if (blockIdx.x == 0 && tid == 0) {
        float tail_sum = 0.0f;
        int tail_start = vec_count * 4;
        for (int j = tail_start; j < n; j++) {
            tail_sum += compute_kldiv_value(log_predictions[j], targets[j]);
        }
        atomicAdd(output, tail_sum);
    }
}

torch::Tensor kl_div_cuda_forward_combined(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    int vec_count = n / 4;
    const int blocks = min((vec_count + threads - 1) / threads, 1024);

    kl_div_kernel_combined<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_combined, "Combined KL divergence forward (CUDA)");
}