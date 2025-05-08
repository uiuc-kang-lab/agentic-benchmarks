#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void smooth_l1_loss_first_pass_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* block_results,
    const int n_elements
) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int stride = gridDim.x * blockDim.x;
    const int lane = tid % warpSize;
    const int wid = tid / warpSize;
    
    float thread_sum = 0.0f;
    
    // Vectorized processing
    const int vec_elements = n_elements / 4;
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);
    
    for (int i = gid; i < vec_elements; i += stride) {
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
    
    // Handle remaining elements
    const int remainder_start = vec_elements * 4;
    for (int i = remainder_start + gid; i < n_elements; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }
    
    // Warp reduction
    thread_sum = warp_reduce(thread_sum);
    
    __shared__ float warp_sums[32];
    if (lane == 0) {
        warp_sums[wid] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction in first warp and store block result
    if (wid == 0) {
        float val = (tid < (blockDim.x + warpSize - 1) / warpSize) ? warp_sums[lane] : 0.0f;
        val = warp_reduce(val);
        if (lane == 0) {
            block_results[blockIdx.x] = val;
        }
    }
}

__global__ void smooth_l1_loss_second_pass_kernel(
    float* block_results,
    float* final_output,
    const int n_blocks,
    const int n_elements
) {
    float sum = 0.0f;
    
    // Each thread processes multiple blocks if necessary
    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        sum += block_results[i];
    }
    
    // Warp reduction
    sum = warp_reduce(sum);
    
    // Only one thread writes final result
    if (threadIdx.x == 0) {
        final_output[0] = sum / n_elements;
    }
}

torch::Tensor smooth_l1_loss_two_pass(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");
    
    const int n_elements = predictions.numel();
    const int block_size = 256;
    const int grid_size = std::min(65535, (n_elements + block_size - 1) / block_size);
    
    // Allocate temporary storage for block results
    auto block_results = torch::empty({grid_size}, predictions.options());
    auto output = torch::zeros({1}, predictions.options());
    
    // First pass: compute partial sums
    smooth_l1_loss_first_pass_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        n_elements
    );
    
    // Second pass: combine partial results
    const int second_block_size = 256;
    smooth_l1_loss_second_pass_kernel<<<1, second_block_size>>>(
        block_results.data_ptr<float>(),
        output.data_ptr<float>(),
        grid_size,
        n_elements
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_two_pass, "Two-pass Smooth L1 Loss (CUDA)");
}