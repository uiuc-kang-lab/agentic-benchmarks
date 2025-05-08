#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template<bool UseVectorization>
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

    if constexpr (UseVectorization) {
        // Process elements in groups of 4 using vectorized loads
        int vec_count = n_elements / 4;
        const float4* predictions4 = reinterpret_cast<const float4*>(predictions);
        const float4* targets4 = reinterpret_cast<const float4*>(targets);
        
        #pragma unroll
        for (int i = idx; i < vec_count; i += stride) {
            float4 p = __ldg(predictions4 + i);
            float4 t = __ldg(targets4 + i);
            
            float diffs[4] = {p.x - t.x, p.y - t.y, p.z - t.z, p.w - t.w};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float abs_diff = fabsf(diffs[j]);
                thread_sum += (abs_diff < 1.0f) ? 0.5f * diffs[j] * diffs[j] : abs_diff - 0.5f;
            }
        }

        // Handle remaining elements
        int start = vec_count * 4;
        for (int i = start + idx; i < n_elements; i += stride) {
            float diff = __ldg(predictions + i) - __ldg(targets + i);
            float abs_diff = fabsf(diff);
            thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
        }
    } else {
        // Non-vectorized version for small input sizes
        for (int i = idx; i < n_elements; i += stride) {
            float diff = __ldg(predictions + i) - __ldg(targets + i);
            float abs_diff = fabsf(diff);
            thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
        }
    }

    // Warp-level reduction first
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Block-level reduction using shared memory (only for the first thread in each warp)
    __shared__ float shared_sum[32]; // Only need warpSize elements now
    int wid = tid / warpSize;
    int lid = tid % warpSize;

    if (lid == 0) shared_sum[wid] = thread_sum;
    __syncthreads();

    // Final reduction with first warp
    if (wid == 0) {
        thread_sum = (lid < (blockDim.x / warpSize)) ? shared_sum[lid] : 0.0f;
        #pragma unroll
        for (int offset = (blockDim.x/warpSize)/2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        if (lid == 0) {
            atomicAdd(output, thread_sum / n_elements);
        }
    }
}

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
    const int grid_size = (n + block_size - 1) / block_size;

    // Choose vectorization based on input size and alignment
    bool use_vectorization = (n >= 1024) && ((size_t)predictions.data_ptr() % 16 == 0) && ((size_t)targets.data_ptr() % 16 == 0);
    
    if (use_vectorization) {
        optimized_smooth_l1_loss_kernel<true><<<grid_size, block_size>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n
        );
    } else {
        optimized_smooth_l1_loss_kernel<false><<<grid_size, block_size>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_smooth_l1_loss_cuda, "Optimized Smooth L1 Loss (CUDA)");
}