#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    // All threads active mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// New combined kernel using vectorized loads and warp-level reductions
__global__ void smooth_l1_loss_fast_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float thread_sum = 0.0f;

    // Vectorized processing using float4
    int vec_count = n_elements / 4;  // number of float4 groups
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);

    for (int i = idx; i < vec_count; i += stride) {
        float4 p = __ldg(&pred4[i]);
        float4 t = __ldg(&targ4[i]);

        float diff = p.x - t.x;
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f * diff * diff : fabsf(diff) - 0.5f;

        diff = p.y - t.y;
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f * diff * diff : fabsf(diff) - 0.5f;

        diff = p.z - t.z;
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f * diff * diff : fabsf(diff) - 0.5f;

        diff = p.w - t.w;
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f * diff * diff : fabsf(diff) - 0.5f;
    }

    // Process remaining elements scalarly
    int scalar_base = vec_count * 4;
    for (int i = scalar_base + idx; i < n_elements; i += stride) {
        float diff = __ldg(&predictions[i]) - __ldg(&targets[i]);
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f * diff * diff : fabsf(diff) - 0.5f;
    }

    // Intra-warp reduction using shuffle
    float sum = warpReduceSum(thread_sum);

    // Each warp writes its result to shared memory
    __shared__ float shared[32];  // assume blockDim.x <= 1024 (i.e., at most 32 warps per block)
    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warpId] = sum;
    }
    __syncthreads();

    // First warp reduces the per-warp partial sums
    float block_sum = 0.0f;
    if (threadIdx.x < (blockDim.x / warpSize)) {
        block_sum = shared[lane];
    }
    if (threadIdx.x < warpSize) {
        block_sum = warpReduceSum(block_sum);
    }

    // Thread 0 atomically adds the block's contribution into global output
    if (threadIdx.x == 0) {
        // Normalize the total loss by the number of elements
        atomicAdd(output, block_sum / n_elements);
    }
}

// Host function
torch::Tensor smooth_l1_loss_fast(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input shape mismatch");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Non-contiguous inputs");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "CUDA tensors required");

    int n_elements = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    int grid_size = (n_elements / 4 + block_size - 1) / block_size;
    if (grid_size < 1) grid_size = 1;

    smooth_l1_loss_fast_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_fast, "Fast Smooth L1 Loss using warp-level reduction");
}
