#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Constant memory for frequently accessed values
__constant__ float THRESHOLD = 1.0f;
__constant__ float HALF = 0.5f;

__global__ void smooth_l1_loss_const_mem_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    const int n_elements
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Pre-cast pointers for vectorized processing
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);
    const int vec_count = n_elements / 4;

    // Vectorized processing using float4
    for (int i = idx; i < vec_count; i += stride) {
        float4 p = __ldg(pred4 + i);
        float4 t = __ldg(targ4 + i);

        // Process x component
        float diff = p.x - t.x;
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < THRESHOLD) ? 
            HALF * diff * diff : 
            abs_diff - HALF;

        // Process y component
        diff = p.y - t.y;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < THRESHOLD) ? 
            HALF * diff * diff : 
            abs_diff - HALF;

        // Process z component
        diff = p.z - t.z;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < THRESHOLD) ? 
            HALF * diff * diff : 
            abs_diff - HALF;

        // Process w component
        diff = p.w - t.w;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < THRESHOLD) ? 
            HALF * diff * diff : 
            abs_diff - HALF;
    }

    // Handle remaining elements
    int scalar_start = vec_count * 4;
    for (int i = scalar_start + idx; i < n_elements; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < THRESHOLD) ? 
            HALF * diff * diff : 
            abs_diff - HALF;
    }

    // Block-level reduction using shared memory
    __shared__ float shared_mem[256];
    shared_mem[tid] = thread_sum;
    __syncthreads();

    // Optimized reduction
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction (last 32 elements)
    if (tid < 32) {
        volatile float* smem = shared_mem;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }

    // Final atomic add
    if (tid == 0) {
        atomicAdd(output, shared_mem[0] / n_elements);
    }
}

torch::Tensor smooth_l1_loss_const_mem(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    int grid_size = (n / 4 + block_size - 1) / block_size;
    grid_size = grid_size > 0 ? grid_size : 1;

    smooth_l1_loss_const_mem_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_const_mem, "Smooth L1 Loss with constant memory optimization");
}