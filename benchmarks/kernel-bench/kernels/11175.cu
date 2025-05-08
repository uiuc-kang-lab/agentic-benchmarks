#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Templated CUDA kernel allowing experimentation with different block sizes
// Uses vectorized memory accesses (via float4 and __ldg) and warp-level reduction
// to compute the Smooth L1 (Huber) Loss and average it over all elements.

template <int BLOCK_SIZE>
__global__ void smooth_l1_loss_kernel_opt(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    int stride = gridDim.x * BLOCK_SIZE;
    float thread_sum = 0.0f;

    // Process data with vectorized loads using float4 when possible
    int vec_count = n_elements / 4;  // Number of groups of 4 elements
    const float4* p4 = reinterpret_cast<const float4*>(predictions);
    const float4* t4 = reinterpret_cast<const float4*>(targets);

    for (int i = idx; i < vec_count; i += stride) {
        float4 p = __ldg(p4 + i);
        float4 t = __ldg(t4 + i);

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

    // Handle any remaining elements that don't fill a complete vectorized group
    int remainder_start = vec_count * 4;
    for (int i = remainder_start + idx; i < n_elements; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Warp-level reduction using shuffle intrinsics
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Each warp's lane 0 writes its result to shared memory
    __shared__ float warp_sums[(BLOCK_SIZE + 31) / 32];
    if ((tid & (warpSize - 1)) == 0) {
        warp_sums[tid / warpSize] = thread_sum;
    }
    __syncthreads();

    // Let the first warp reduce the results from each warp
    float block_sum = 0.0f;
    int numWarps = (BLOCK_SIZE + 31) / 32;
    if (tid < numWarps) {
        block_sum = warp_sums[tid];
    } else {
        block_sum = 0.0f;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    }

    // The first thread atomically adds the block's contribution (averaged) to the output
    if (tid == 0) {
        atomicAdd(output, block_sum / n_elements);
    }
}

// Host function allowing experimentation with different block sizes.
// The block_size parameter should be one of: 32, 64, 128, 256, or 512.

torch::Tensor smooth_l1_loss_optimal(
    torch::Tensor predictions,
    torch::Tensor targets,
    int block_size
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

    int n_elements = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    // Calculate grid size based on the vectorized load count
    int vec_count = n_elements / 4;
    int grid_size = (vec_count + block_size - 1) / block_size;
    if (grid_size < 1) grid_size = 1;

    // Launch the templated kernel with the chosen block size
    if (block_size == 32) {
        smooth_l1_loss_kernel_opt<32><<<grid_size, 32>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n_elements
        );
    } else if (block_size == 64) {
        smooth_l1_loss_kernel_opt<64><<<grid_size, 64>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n_elements
        );
    } else if (block_size == 128) {
        smooth_l1_loss_kernel_opt<128><<<grid_size, 128>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n_elements
        );
    } else if (block_size == 256) {
        smooth_l1_loss_kernel_opt<256><<<grid_size, 256>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n_elements
        );
    } else if (block_size == 512) {
        smooth_l1_loss_kernel_opt<512><<<grid_size, 512>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n_elements
        );
    } else {
        // Default to block size 256 if an unsupported value is provided
        smooth_l1_loss_kernel_opt<256><<<grid_size, 256>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n_elements
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_optimal, "Smooth L1 Loss (CUDA) with optimal block size selection");
}
