#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel that computes Smooth L1 (Huber) Loss using vectorized float4 loads and warp-level reductions with __shfl_down_sync.
// Instead of using shared memory for intra-block reductions, each warp performs its own reduction and then its lane 0 atomically adds its value to the global output.

__global__ void smooth_l1_loss_kernel_warp_atomic(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Vectorized processing using float4 loads for 128-bit aligned memory access
    int vec_count = n_elements / 4;  // number of float4 groups
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

    // Process remaining elements that don't fit into a vectorized load
    int scalar_start = vec_count * 4;
    for (int i = scalar_start + idx; i < n_elements; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Each warp's lane 0 atomically adds its partial sum to the global output.
    if ((tid & (warpSize - 1)) == 0) {
        atomicAdd(output, thread_sum / n_elements);
    }
}

// Host function to launch the kernel
torch::Tensor smooth_l1_loss_cuda_warp_atomic(
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
    int grid_size = (vec_count + block_size - 1) / block_size;
    if (grid_size < 1) grid_size = 1;

    smooth_l1_loss_kernel_warp_atomic<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda_warp_atomic, "Smooth L1 Loss (CUDA) using vectorized loads and warp-level atomic reductions");
}
