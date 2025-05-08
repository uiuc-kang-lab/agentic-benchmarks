#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel: Smooth L1 (Huber) Loss using __ldg() and 128-bit aligned loads via float4
__global__ void smooth_l1_loss_kernel_aligned(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Process bulk in 128-bit (float4) chunks. Assumes predictions and targets are 16-byte aligned.
    int n_vec = n_elements / 4;  // number of float4 loads
    const float4* preds_vec = reinterpret_cast<const float4*>(predictions);
    const float4* targets_vec = reinterpret_cast<const float4*>(targets);
    
    for (int i = idx; i < n_vec; i += stride) {
        // Use __ldg to load from global memory through the read-only cache
        float4 p = __ldg(&preds_vec[i]);
        float4 t = __ldg(&targets_vec[i]);
        
        // Compute differences for 4 elements
        float diff0 = p.x - t.x;
        float diff1 = p.y - t.y;
        float diff2 = p.z - t.z;
        float diff3 = p.w - t.w;
        
        float abs0 = fabsf(diff0);
        float abs1 = fabsf(diff1);
        float abs2 = fabsf(diff2);
        float abs3 = fabsf(diff3);
        
        thread_sum += (abs0 < 1.0f) ? 0.5f * diff0 * diff0 : abs0 - 0.5f;
        thread_sum += (abs1 < 1.0f) ? 0.5f * diff1 * diff1 : abs1 - 0.5f;
        thread_sum += (abs2 < 1.0f) ? 0.5f * diff2 * diff2 : abs2 - 0.5f;
        thread_sum += (abs3 < 1.0f) ? 0.5f * diff3 * diff3 : abs3 - 0.5f;
    }
    
    // Process remaining elements not fitting into float4
    int vec_end = n_vec * 4;
    for (int i = vec_end + idx; i < n_elements; i += stride) {
        float pred = __ldg(&predictions[i]);
        float targ = __ldg(&targets[i]);
        float diff = pred - targ;
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }
    
    // Block-level reduction using shared memory (dynamically allocated)
    extern __shared__ float shared_sum[];
    int tid = threadIdx.x;
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // Averaging the loss
        atomicAdd(output, shared_sum[0] / n_elements);
    }
}

// Host wrapper for the CUDA kernel
torch::Tensor smooth_l1_loss_cuda_aligned(
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
    const int grid_size = (n + block_size - 1) / block_size;
    
    // Launch kernel with dynamic shared memory allocation
    smooth_l1_loss_kernel_aligned<<<grid_size, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda_aligned, "Smooth L1 Loss with __ldg and 128-bit aligned loads (CUDA)");
}
