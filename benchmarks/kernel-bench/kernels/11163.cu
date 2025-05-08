#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for Smooth L1 Loss (Huber Loss) using __ldg() for optimized global memory loads
// and vectorized accesses with float4 for 128-bit alignment.
__global__ void smooth_l1_loss_kernel_ldg(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float thread_sum = 0.0f;

    // Process elements in groups of 4 using vectorized loads (128-bit aligned) if possible
    int vec_count = n_elements / 4;  // number of float4 groups
    
    for (int i = idx; i < vec_count; i += stride) {
        // Cast pointers to float4 for 128-bit loads and use __ldg() to leverage read-only cache
        const float4* predictions4 = reinterpret_cast<const float4*>(predictions);
        const float4* targets4 = reinterpret_cast<const float4*>(targets);
        float4 p = __ldg(predictions4 + i);
        float4 t = __ldg(targets4 + i);
        
        // Process each of the 4 components
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

    // Handle any remaining elements that don't fit into a group of 4
    int start = vec_count * 4;
    for (int i = start + idx; i < n_elements; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Block-level reduction using shared memory
    __shared__ float shared_sum[256];
    shared_sum[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Atomic add the average loss computed from this block
        atomicAdd(output, shared_sum[0] / n_elements);
    }
}

// Host function wrapper
torch::Tensor smooth_l1_loss_cuda_ldg(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    int vec_count = n / 4;  // number of groups of 4
    int grid_size = (vec_count > 0) ? ((vec_count + block_size - 1) / block_size) : 1;

    smooth_l1_loss_kernel_ldg<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda_ldg, "Smooth L1 Loss (CUDA) with __ldg() and 128-bit aligned accesses");
}
