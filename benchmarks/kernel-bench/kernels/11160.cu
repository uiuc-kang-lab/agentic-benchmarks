#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel using 2D grid and 2D block mapping
__global__ void smooth_l1_loss_kernel_2d(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    const int n_elements
) {
    // Compute global thread coordinates in 2D
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute the pitch (total threads per row in the grid)
    int pitch = gridDim.x * blockDim.x;
    // Linear index computed from 2D coordinates
    int idx = global_y * pitch + global_x;

    // Total number of threads in the entire grid
    int total_threads = pitch * (gridDim.y * blockDim.y);

    float thread_sum = 0.0f;

    // Use a grid-stride loop to cover all elements
    for (int i = idx; i < n_elements; i += total_threads) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        if (abs_diff < 1.0f) {
            thread_sum += 0.5f * diff * diff;
        } else {
            thread_sum += abs_diff - 0.5f;
        }
    }

    // Intra-block reduction using shared memory
    // Flatten 2D block index into a 1D local index
    int local_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Declare dynamic shared memory array
    extern __shared__ float shared_sum[];
    shared_sum[local_idx] = thread_sum;
    __syncthreads();

    int block_threads = blockDim.x * blockDim.y;
    // Reduction loop (simple tree-based reduction)
    for (int s = block_threads / 2; s > 0; s >>= 1) {
        if (local_idx < s) {
            shared_sum[local_idx] += shared_sum[local_idx + s];
        }
        __syncthreads();
    }

    // The first thread in the block atomically adds this block's contribution
    if (local_idx == 0) {
        atomicAdd(output, shared_sum[0] / n_elements);
    }
}

// Host function
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

    int n_elements = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    // Define 2D block dimensions (e.g., 16x16 = 256 threads per block)
    dim3 block(16, 16);
    int threads_per_block = block.x * block.y; // 256

    // Compute total number of blocks required (1D count)
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;
    
    // Map the 1D block count into a 2D grid
    int grid_x = static_cast<int>(ceil(sqrt(static_cast<double>(num_blocks))));
    int grid_y = (num_blocks + grid_x - 1) / grid_x;
    dim3 grid(grid_x, grid_y);

    // Allocate dynamic shared memory (one float per thread in a block)
    size_t shared_mem_size = threads_per_block * sizeof(float);

    smooth_l1_loss_kernel_2d<<<grid, block, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Smooth L1 Loss (CUDA) with 2D grid and block mapping");
}
