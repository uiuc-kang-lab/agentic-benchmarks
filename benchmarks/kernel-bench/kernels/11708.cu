#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// CUDA kernel with vectorized loads for coalesced global memory access
__global__ void vectorized_coalesced_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Each thread processes multiple elements via grid-stride loops
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float thread_sum = 0.0f;

    // Number of elements that can be loaded as float4 vectors
    int n4 = n / 4;
    const float4* log4 = reinterpret_cast<const float4*>(log_predictions);
    const float4* tgt4 = reinterpret_cast<const float4*>(targets);

    // Process vectorized loads, ensuring coalesced access
    for (int i = tid; i < n4; i += stride) {
        float4 logv = log4[i];
        float4 tgtv = tgt4[i];
        thread_sum += expf(logv.x) - tgtv.x * logv.x;
        thread_sum += expf(logv.y) - tgtv.y * logv.y;
        thread_sum += expf(logv.z) - tgtv.z * logv.z;
        thread_sum += expf(logv.w) - tgtv.w * logv.w;
    }

    // Process remaining tail elements
    int vec_elems = n4 * 4;
    for (int i = vec_elems + tid; i < n; i += stride) {
        float log_val = log_predictions[i];
        float tgt_val = targets[i];
        thread_sum += expf(log_val) - tgt_val * log_val;
    }

    // Warp-level reduction using shuffle down intrinsics
    const int warpSize = 32;
    int lane = threadIdx.x % warpSize;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Allocate shared memory for warp sums
    extern __shared__ float shared_sum[];
    if (lane == 0) {
        shared_sum[threadIdx.x / warpSize] = thread_sum;
    }
    __syncthreads();

    // Let the first warp reduce the block's sums
    int numWarps = blockDim.x / warpSize;
    if (threadIdx.x < warpSize) {
        float block_sum = (threadIdx.x < numWarps) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// Host function that sets up the kernel launch parameters and invokes the kernel

torch::Tensor vectorized_coalesced_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Dynamic block size selection based on problem size
    int block_size = 256;
    if (n > 65536) block_size = 512;
    else if (n < 8192) block_size = 128;

    const int max_blocks = 256;
    int blocks = std::min(max_blocks, (n + block_size - 1) / block_size);

    // Shared memory size: one float per warp
    int num_warps = block_size / 32;
    int shared_mem = num_warps * sizeof(float);

    vectorized_coalesced_kl_div_kernel<<<blocks, block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &vectorized_coalesced_kl_div_forward, "KLDivLoss with vectorized, coalesced global memory accesses (CUDA)");
}
