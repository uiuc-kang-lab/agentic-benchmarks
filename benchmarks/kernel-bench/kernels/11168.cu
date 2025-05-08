#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void smooth_l1_loss_optimized_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Vectorized processing
    int vec_count = n_elements / 4;
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);

    for (int i = idx; i < vec_count; i += stride) {
        float4 p = __ldg(pred4 + i);
        float4 t = __ldg(targ4 + i);

        float diff = p.x - t.x;
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f*diff*diff : fabsf(diff)-0.5f;
        
        diff = p.y - t.y;
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f*diff*diff : fabsf(diff)-0.5f;
        
        diff = p.z - t.z;
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f*diff*diff : fabsf(diff)-0.5f;
        
        diff = p.w - t.w;
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f*diff*diff : fabsf(diff)-0.5f;
    }

    // Scalar processing for remainder
    int scalar_base = vec_count * 4;
    for (int i = scalar_base + idx; i < n_elements; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f*diff*diff : fabsf(diff)-0.5f;
    }

    // Optimized block reduction
    __shared__ float shared_mem[256];
    shared_mem[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared_mem[0] / n_elements);
    }
}

torch::Tensor smooth_l1_loss_optimized(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input shape mismatch");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Non-contiguous inputs");

    int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    int grid_size = (n / 4 + block_size - 1) / block_size;
    grid_size = grid_size > 0 ? grid_size : 1;

    smooth_l1_loss_optimized_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_optimized, "Optimized Smooth L1 Loss");
}