#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void hinge_loss_warp_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    float* __restrict__ partial_sums,
    const int n
) {
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int gid = blockIdx.x * BLOCK_SIZE + tid;
    const int grid_stride = gridDim.x * BLOCK_SIZE;
    
    float thread_sum = 0.0f;

    // Process elements with grid stride loop
    for (int idx = gid; idx < n; idx += grid_stride) {
        const float pred = __ldg(&predictions[idx]);
        const float target = __ldg(&targets[idx]);
        const float loss = fmaxf(0.0f, 1.0f - pred * target);
        output[idx] = loss;
        thread_sum += loss;
    }

    // Warp-level reduction
    thread_sum = warpReduceSum(thread_sum);

    // First thread in each warp writes the result
    if (lane == 0) {
        partial_sums[blockIdx.x * WARPS_PER_BLOCK + wid] = thread_sum;
    }
}

__global__ void final_reduction_kernel(
    float* __restrict__ partial_sums,
    float* __restrict__ final_result,
    const int n_sums
) {
    float sum = 0.0f;
    const int tid = threadIdx.x;
    
    for (int i = tid; i < n_sums; i += WARP_SIZE) {
        sum += partial_sums[i];
    }
    
    sum = warpReduceSum(sum);
    
    if (tid == 0) {
        *final_result = sum / n_sums;
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    const int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);
    
    const int blocks = min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);
    const int n_partial_sums = blocks * WARPS_PER_BLOCK;
    
    auto partial_sums = torch::empty({n_partial_sums}, predictions.options());
    auto final_result = torch::empty({1}, predictions.options());

    hinge_loss_warp_kernel<<<blocks, BLOCK_SIZE>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        n
    );

    final_reduction_kernel<<<1, WARP_SIZE>>>(
        partial_sums.data_ptr<float>(),
        final_result.data_ptr<float>(),
        n_partial_sums
    );

    return final_result[0];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp Optimized Hinge Loss Forward");
}