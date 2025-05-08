#include <torch/extension.h>
#include <cmath>
#include <cfloat>

// This kernel uses a 2D block structure to process multiple samples per block.
// The x-dimension of the block is used to perform reduction across the num_classes dimension,
// while the y-dimension indexes different samples within the same block. This mapping improves occupancy
// and ensures that thread and block indexing match the problem domain efficiently.

// Kernel: each block processes 'blockDim.y' samples. For each sample, threads in the x-direction load and reduce
// the logits to compute the maximum value and the sum of exponentials for numerical stability.

__global__ void cross_entropy_loss_kernel_2d(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    // Declare shared memory: one row per sample in the block (blockDim.y rows, each of blockDim.x elements)
    extern __shared__ float sdata[];
    
    // Each thread block processes several samples in the batch along the y-dimension.
    int sample = blockIdx.x * blockDim.y + threadIdx.y;
    if (sample >= batch_size) return;

    // Pointer to the beginning of the logits for the given sample
    const float* sample_logits = logits + sample * num_classes;

    // Each row in shared memory is used to reduce over the num_classes for one sample.
    // Compute the offset for this sample's row in shared memory
    int tid = threadIdx.x;
    float* srow = sdata + threadIdx.y * blockDim.x;

    // Phase 1: Compute the maximum logit for numerical stability
    float local_max = -FLT_MAX;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        float val = sample_logits[j];
        local_max = fmaxf(local_max, val);
    }
    srow[tid] = local_max;
    __syncthreads();

    // Reduction: compute maximum across the block's x-dimension for this sample
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            srow[tid] = fmaxf(srow[tid], srow[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = srow[0];

    // Phase 2: Compute the sum of exponentials using the computed maximum
    float local_sum = 0.0f;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        local_sum += expf(sample_logits[j] - max_val);
    }
    srow[tid] = local_sum;
    __syncthreads();
    
    // Reduction: sum the partial sums along the x-dimension
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            srow[tid] += srow[tid + stride];
        }
        __syncthreads();
    }
    float sum_exp = srow[0];

    // Thread 0 in the sample row computes and writes the final loss
    if (tid == 0) {
        int64_t target = targets[sample];
        float loss = - (sample_logits[target] - max_val - logf(sum_exp));
        losses[sample] = loss;
    }
}

// Host function: sets up kernel launch parameters with a 2D thread block

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    // Input checks
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be a Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be an Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    // Allocate output tensor for per-sample losses
    auto losses = torch::empty({batch_size}, predictions.options());

    // Define 2D block dimensions: blockDim.x handles reduction over classes, blockDim.y processes multiple samples
    // Adjust these parameters as needed for optimal occupancy
    const int threads_x = 128;  // threads for reduction over classes
    const int threads_y = 4;    // samples per block
    dim3 block(threads_x, threads_y);
    // Grid: each block processes 'threads_y' samples
    int grid_x = (batch_size + threads_y - 1) / threads_y;
    dim3 grid(grid_x);

    // Shared memory size: blockDim.x * blockDim.y floats
    size_t shared_mem_size = threads_x * threads_y * sizeof(float);

    cross_entropy_loss_kernel_2d<<<grid, block, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_2d: ", cudaGetErrorString(err));

    // Compute mean loss over batch
    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 95_CrossEntropyLoss forward (CUDA) with 2D thread block mapping");
}
