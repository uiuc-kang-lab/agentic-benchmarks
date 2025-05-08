#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    // Shared memory for tiling
    __shared__ float shared_pred[1024];  // 512 * 2 for double buffering
    __shared__ float shared_targ[1024];  // 512 * 2 for double buffering
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Process elements using shared memory tiles
    const int TILE_SIZE = 512;
    for (int base = blockIdx.x * blockDim.x * 4; base < n_elements; base += stride * 4) {
        // Load data into shared memory
        for (int offset = 0; offset < 4 && (base + tid * 4 + offset) < n_elements; offset++) {
            int load_idx = base + tid * 4 + offset;
            if (load_idx < n_elements) {
                shared_pred[tid * 4 + offset] = predictions[load_idx];
                shared_targ[tid * 4 + offset] = targets[load_idx];
            }
        }
        __syncthreads();

        // Process data from shared memory
        for (int i = 0; i < 4 && (base + tid * 4 + i) < n_elements; i++) {
            float diff = shared_pred[tid * 4 + i] - shared_targ[tid * 4 + i];
            float abs_diff = fabsf(diff);
            thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
        }
        __syncthreads();
    }

    // Block-wise reduction
    __shared__ float shared_sum[512]; // Adjusted shared memory to match max block size
    int tid = threadIdx.x;
    shared_sum[tid] = thread_sum;
    __syncthreads();

    // Unroll loop and reduce
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

    const int block_size = 512; // Experimented optimal block size
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
