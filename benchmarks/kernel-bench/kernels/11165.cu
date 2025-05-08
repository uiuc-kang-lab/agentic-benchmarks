#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle instructions
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void optimized_smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float thread_sum = 0.0f;

    // Vectorized processing using float4 for 128-bit alignment
    int vec_count = n_elements / 4;
    const float4* predictions4 = reinterpret_cast<const float4*>(predictions);
    const float4* targets4 = reinterpret_cast<const float4*>(targets);

    for (int i = idx; i < vec_count; i += stride) {
        float4 p = __ldg(predictions4 + i);
        float4 t = __ldg(targets4 + i);

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

    // Handle remaining elements
    int start = vec_count * 4;
    for (int i = start + idx; i < n_elements; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Intra-warp reduction
    thread_sum = warp_reduce_sum(thread_sum);

    // Shared memory for warp-level sums
    __shared__ float shared[32];
    int lane = tid & (warpSize - 1);
    int warpId = tid / warpSize;

    if (lane == 0) {
        shared[warpId] = thread_sum;
    }
    __syncthreads();

    // First warp reduces the warp results
    float block_sum = 0.0f;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (tid < numWarps) {
        block_sum = shared[tid];
    }
    if (tid < warpSize) {
        block_sum = warp_reduce_sum(block_sum);
    }

    // Atomic add to output
    if (tid == 0) {
        atomicAdd(output, block_sum / n_elements);
    }
}

// Host function wrapper
torch::Tensor optimized_smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    int vec_count = n / 4;
    int grid_size = (vec_count > 0) ? ((vec_count + block_size - 1) / block_size) : 1;

    optimized_smooth_l1_loss_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_smooth_l1_loss_cuda, "Optimized Smooth L1 Loss (CUDA) with vectorized loads and warp-level reduction");
}
