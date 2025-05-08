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
    const int gridSize = gridDim.x * blockDim.x;
    const int stride = blockSize * gridSize;
    const int gid = bid * blockSize + tid;
    
    __shared__ float shared_sum[512];
    float thread_sum = 0.0f;
    
    // Process multiple elements per thread using grid-stride loop
    for (int idx = gid; idx < n_elements; idx += stride) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabsf(diff);
        if (abs_diff < 1.0f) {
            thread_sum += 0.5f * diff * diff;
        } else {
            thread_sum += abs_diff - 0.5f;
        }
    }
    
    // Store in shared memory
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockSize/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
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

    const int block_size = 512;
    const int num_blocks = min(65535, (n + block_size - 1) / block_size);

    smooth_l1_loss_kernel<<<num_blocks, block_size>>>(
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