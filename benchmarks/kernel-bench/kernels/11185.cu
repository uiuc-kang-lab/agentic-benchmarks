#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle intrinsics without extra synchronizations
__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA kernel for Smooth L1 Loss using vectorized loads and warp-level reduction
// This version avoids unnecessary __syncthreads() by letting each warp's leader perform an atomic add
// after reducing its lane values. This minimizes synchronization overhead.
__global__ void smooth_l1_loss_warp_atomic_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    const int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float thread_sum = 0.0f;

    // Process data in groups of 4 (vectorized access)
    int vec_count = n_elements / 4;
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);

    for (int i = idx; i < vec_count; i += stride) {
        float4 p = __ldg(pred4 + i);
        float4 t = __ldg(targ4 + i);
        
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

    // Process any remaining elements with scalar loads
    int remainder_start = vec_count * 4;
    for (int i = remainder_start + idx; i < n_elements; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Perform warp-level reduction; no __syncthreads() needed within a warp
    int lane = threadIdx.x & (warpSize - 1);
    thread_sum = warp_reduce(thread_sum);

    // Each warp leader (lane 0) atomically adds its reduced sum to the global output
    if (lane == 0) {
        atomicAdd(output, thread_sum / n_elements);
    }
}

// Host function wrapping the CUDA kernel
torch::Tensor smooth_l1_loss_warp_atomic(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(
        predictions.sizes() == targets.sizes(),
        "Input tensors must have the same shape"
    );
    TORCH_CHECK(
        predictions.is_contiguous() && targets.is_contiguous(),
        "Input tensors must be contiguous"
    );
    TORCH_CHECK(
        predictions.device().is_cuda() && targets.device().is_cuda(),
        "Inputs must be CUDA tensors"
    );

    int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    smooth_l1_loss_warp_atomic_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_warp_atomic, "Smooth L1 Loss (CUDA) with warp atomic reduction and minimal synchronization");
}
