#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef float float4_t __attribute__((ext_vector_type(4)));

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    const int vector_stride = gridDim.x * blockDim.x * 4;
    const int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float thread_sum = 0.0f;

    // Process 4 elements at a time using vector loads
    for (int i = base_idx; i < n_elements - 3; i += vector_stride) {
        float4_t pred_vec = *reinterpret_cast<const float4_t*>(&predictions[i]);
        float4_t targ_vec = *reinterpret_cast<const float4_t*>(&targets[i]);
        
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            float diff = pred_vec[j] - targ_vec[j];
            float abs_diff = fabsf(diff);
            thread_sum += (abs_diff < 1.0f) ? (0.5f * diff * diff) : (abs_diff - 0.5f);
        }
    }

    // Handle remaining elements
    for (int i = base_idx + 4 * (n_elements/4); i < n_elements; i++) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? (0.5f * diff * diff) : (abs_diff - 0.5f);
    }

    // Warp-level reduction using tree parallel method
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

    // First thread in warp accumulates to global memory
    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, thread_sum / n_elements);
    }
}

torch::Tensor smooth_l1_loss_cuda(
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
    const int grid_size = min(65535, (n + block_size - 1) / block_size);

    smooth_l1_loss_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Smooth L1 Loss (CUDA)");
}
