#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for Smooth L1 Loss (Huber Loss) with manual loop unrolling
__global__ void smooth_l1_loss_kernel_unrolled(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Unroll factor for the main loop
    const int unroll_factor = 4;
    
    // Main loop with manual unrolling
    for (int i = idx; i < n_elements; i += stride * unroll_factor) {
        #pragma unroll
        for (int j = 0; j < unroll_factor; j++) {
            int index = i + j * stride;
            if (index < n_elements) {
                float diff = predictions[index] - targets[index];
                float abs_diff = fabsf(diff);
                thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
            }
        }
    }

    // Reduction in shared memory
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    shared_sum[tid] = thread_sum;
    __syncthreads();

    #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared_sum[0] / n_elements);
    }
}

// Host function wrapper
torch::Tensor smooth_l1_loss_cuda_unrolled(
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

    smooth_l1_loss_kernel_unrolled<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda_unrolled, "Smooth L1 Loss (CUDA) with manual loop unrolling");
}
