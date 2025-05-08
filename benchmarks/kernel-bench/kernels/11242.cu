#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int blockSize = blockDim.x;
    const int gid = bid * blockSize + tid;
    const int stride = gridDim.x * blockDim.x * 4;  // Increased stride for unrolling
    
    float thread_sum = 0.0f;
    
    // Manual loop unrolling with 4 elements per iteration
    for (int i = gid * 4; i < n_elements; i += stride) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (i + j < n_elements) {
                const float diff = predictions[i + j] - targets[i + j];
                const float abs_diff = fabsf(diff);
                thread_sum += (abs_diff < 1.0f) ? 
                    (0.5f * diff * diff) : 
                    (abs_diff - 0.5f);
            }
        }
    }
    
    __shared__ float shared_sum[256];
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Explicitly unrolled reduction
    if (tid < 128) shared_sum[tid] += shared_sum[tid + 128];
    __syncthreads();
    if (tid < 64) shared_sum[tid] += shared_sum[tid + 64];
    __syncthreads();
    
    // Last warp reduction unrolled
    if (tid < 32) {
        volatile float* smem = shared_sum;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) {
        atomicAdd(output, shared_sum[0] / n_elements);
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
    const int grid_size = (n + block_size * 4 - 1) / (block_size * 4);

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