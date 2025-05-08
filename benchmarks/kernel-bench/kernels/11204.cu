#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel computing Smooth L1 (Huber) Loss with vectorized loads and dynamic shared memory reduction
__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Process 4 elements per thread via loop unrolling
    for (int i = idx * 4; i < n_elements - 3; i += stride * 4) {
        float diff0 = predictions[i] - targets[i];
        float diff1 = predictions[i + 1] - targets[i + 1];
        float diff2 = predictions[i + 2] - targets[i + 2];
        float diff3 = predictions[i + 3] - targets[i + 3];

        float abs_diff0 = fabsf(diff0);
        float abs_diff1 = fabsf(diff1);
        float abs_diff2 = fabsf(diff2);
        float abs_diff3 = fabsf(diff3);

        thread_sum += (abs_diff0 < 1.0f) ? 0.5f * diff0 * diff0 : abs_diff0 - 0.5f;
        thread_sum += (abs_diff1 < 1.0f) ? 0.5f * diff1 * diff1 : abs_diff1 - 0.5f;
        thread_sum += (abs_diff2 < 1.0f) ? 0.5f * diff2 * diff2 : abs_diff2 - 0.5f;
        thread_sum += (abs_diff3 < 1.0f) ? 0.5f * diff3 * diff3 : abs_diff3 - 0.5f;
    }

    // Process any remaining elements
    int rem_start = (n_elements / 4) * 4;  // largest multiple of 4 below n_elements
    for (int i = idx * 4 + rem_start; i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Block-level reduction using dynamically allocated shared memory
    extern __shared__ float shared_sum[];
    int tid = threadIdx.x;
    shared_sum[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // Atomically add the average loss from this block to the global output
    if (tid == 0) {
        atomicAdd(output, shared_sum[0] / n_elements);
    }
}


// Host function: dynamically selects optimal block size using occupancy and launches the kernel
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

    // Dynamically determine the optimal block size based on occupancy
    int block_size;
    int min_grid_size;
    cudaError_t occ_err = cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size, smooth_l1_loss_kernel, 0, n
    );
    TORCH_CHECK(occ_err == cudaSuccess, "cudaOccupancyMaxPotentialBlockSize failed");

    // Compute grid size; each thread works on 4 elements
    int grid_size = (n + block_size * 4 - 1) / (block_size * 4);

    smooth_l1_loss_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
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
