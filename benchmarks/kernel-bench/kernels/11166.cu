// Combined CUDA kernel for Smooth L1 Loss with vectorized memory accesses and warp-level reduction

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle instructions
__inline__ __device__ float warp_reduce_sum(float val) {
    // Use full warp mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Combined kernel that processes data using float4 vectorized loads and then reduces using warp shuffle intrinsics
__global__ void smooth_l1_loss_kernel_vectorized_warp(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    // Global thread index and stride
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float thread_sum = 0.0f;

    // Process data in groups of 4 elements using float4 for vectorized (128-bit) loads
    int vec_count = n_elements / 4;  // number of float4 groups
    const float4* predictions4 = reinterpret_cast<const float4*>(predictions);
    const float4* targets4 = reinterpret_cast<const float4*>(targets);

    for (int i = idx; i < vec_count; i += stride) {
        float4 p = __ldg(predictions4 + i);
        float4 t = __ldg(targets4 + i);

        // Process each of the four components
        float diff = p.x - t.x;
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        diff = p.y - t.y;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        diff = p.z - t.z;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        diff = p.w - t.w;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Process any remaining elements that don't fill a complete vectorized group
    int remainder_start = vec_count * 4;
    for (int i = remainder_start + idx; i < n_elements; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Perform warp-level reduction using shuffle intrinsics
    thread_sum = warp_reduce_sum(thread_sum);

    // Each warp's lane 0 writes its partial sum to shared memory
    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x / warpSize;
    __shared__ float warp_sums[32];  // enough for up to 1024 threads per block (32 warps)
    if (lane == 0) {
        warp_sums[warpId] = thread_sum;
    }
    __syncthreads();

    // Let the first warp reduce the warp-level sums
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    float block_sum = (threadIdx.x < numWarps) ? warp_sums[threadIdx.x] : 0.0f;
    if (threadIdx.x < warpSize) {
        block_sum = warp_reduce_sum(block_sum);
    }

    // The first thread atomically adds the block's contribution to the output (averaged over n_elements)
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum / n_elements);
    }
}

// Host function wrapping the combined kernel
torch::Tensor smooth_l1_loss_cuda_vectorized_warp(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    int n_elements = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    int vec_count = n_elements / 4;  // for vectorized processing
    int grid_size = (vec_count > 0) ? ((vec_count + block_size - 1) / block_size) : 1;

    smooth_l1_loss_kernel_vectorized_warp<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda_vectorized_warp, "Smooth L1 Loss (CUDA) with vectorized loads and warp-level reduction");
}
