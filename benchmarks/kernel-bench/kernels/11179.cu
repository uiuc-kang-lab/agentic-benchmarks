#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void smooth_l1_loss_strided_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    const int n_elements,
    const int vec_elements
) {
    const int tid = threadIdx.x;
    const int block_stride = blockDim.x * gridDim.x;
    const int global_start = blockIdx.x * blockDim.x + tid;
    
    float thread_sum = 0.0f;
    
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);
    
    #pragma unroll 4
    for (int i = global_start; i < vec_elements; i += block_stride) {
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
    
    const int remainder_start = vec_elements * 4;
    #pragma unroll 4
    for (int i = remainder_start + global_start; i < n_elements; i += block_stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }
    
    __shared__ float smem[256];
    smem[tid] = thread_sum;
    __syncthreads();
    
    if (tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();
    if (tid < 64) smem[tid] += smem[tid + 64];
    __syncthreads();
    
    if (tid < 32) {
        volatile float* vmem = smem;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    
    if (tid == 0) {
        atomicAdd(output, smem[0] / n_elements);
    }
}

torch::Tensor smooth_l1_loss_strided(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");
    
    const int n_elements = predictions.numel();
    const int vec_elements = n_elements / 4;
    auto output = torch::zeros({1}, predictions.options());
    
    const int block_size = 256;
    const int min_elements_per_thread = 4;
    const int target_thread_count = (vec_elements + min_elements_per_thread - 1) / min_elements_per_thread;
    const int grid_size = std::min(
        (target_thread_count + block_size - 1) / block_size,
        32768
    );
    
    smooth_l1_loss_strided_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements,
        vec_elements
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_strided, "Strided Smooth L1 Loss (CUDA)");
}