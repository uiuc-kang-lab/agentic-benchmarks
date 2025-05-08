#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define block size and unroll factor for tiling
#define BLOCK_SIZE 256
#define UNROLL_FACTOR 4
#define TILE_SIZE (BLOCK_SIZE * UNROLL_FACTOR)  // 1024 for BLOCK_SIZE=256 and UNROLL_FACTOR=4

// CUDA kernel that leverages shared memory tiling to reduce global memory latency
__global__ void smooth_l1_loss_shared_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    // Allocate shared memory for a tile of predictions and targets
    __shared__ float s_pred[TILE_SIZE];
    __shared__ float s_targ[TILE_SIZE];

    float thread_sum = 0.0f;

    // Each block processes multiple tiles in a strided fashion
    // Base index for the tile assigned to this block in each iteration
    for (int base = blockIdx.x * TILE_SIZE; base < n_elements; base += gridDim.x * TILE_SIZE) {
        // Determine how many elements to load in this tile
        int tile_elems = ((base + TILE_SIZE) < n_elements) ? TILE_SIZE : (n_elements - base);

        // Load tile data from global memory into shared memory
        for (int i = threadIdx.x; i < tile_elems; i += BLOCK_SIZE) {
            s_pred[i] = predictions[base + i];
            s_targ[i] = targets[base + i];
        }
        __syncthreads();

        // Each thread processes a subset of the tile from shared memory
        for (int i = threadIdx.x; i < tile_elems; i += BLOCK_SIZE) {
            float diff = s_pred[i] - s_targ[i];
            float abs_diff = fabsf(diff);
            // Smooth L1 (Huber) loss: branchless formulation using ternary operator
            float loss = (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
            thread_sum += loss;
        }
        __syncthreads();
    }

    // Reduce the per-thread partial sums within the block using shared memory reduction
    __shared__ float shared_sum[BLOCK_SIZE];
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 atomically adds the block's contribution to the global output (averaged over n_elements)
    if (threadIdx.x == 0) {
        atomicAdd(output, shared_sum[0] / n_elements);
    }
}

// Host function to launch the CUDA kernel using torch extension API
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

    // Compute grid size based on the total number of elements and tile size
    int grid_size = (n + TILE_SIZE - 1) / TILE_SIZE;

    // Launch the kernel
    smooth_l1_loss_shared_kernel<<<grid_size, BLOCK_SIZE>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Smooth L1 Loss using shared memory tiling (CUDA)");
}
