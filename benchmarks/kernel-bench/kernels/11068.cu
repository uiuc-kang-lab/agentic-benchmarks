#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

// This kernel uses a 2D block configuration where each block processes several samples concurrently.
// The threads along the x-dimension collaborate via shared memory to reduce over the classes for each sample.
// This balanced workload distribution minimizes idle threads and bottlenecks across blocks.

__global__ void balanced_cross_entropy_loss_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    // Each block processes blockDim.y samples. 
    // threadIdx.x is used for reduction across classes, threadIdx.y indexes the sample within the block.
    int sample_in_block = threadIdx.y;
    int sample_index = blockIdx.x * blockDim.y + sample_in_block;

    if (sample_index >= batch_size) return;

    // Pointer to the logits for the current sample.
    const float* sample_logits = logits + sample_index * num_classes;

    // Allocate dynamic shared memory:
    // We use two arrays in shared memory: one for max reduction and one for sum reduction.
    extern __shared__ float sdata[];   // total size: 2 * (blockDim.x * blockDim.y) floats
    float* s_max = sdata;              // for max values
    float* s_sum = sdata + blockDim.x * blockDim.y; // for sum of exponentials

    // Compute a local maximum over a subset of classes for this sample.
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        float val = sample_logits[j];
        local_max = fmaxf(local_max, val);
    }

    // Each thread writes its local maximum to shared memory.
    int s_index = sample_in_block * blockDim.x + threadIdx.x;
    s_max[s_index] = local_max;
    __syncthreads();

    // Perform reduction along the x-dimension to compute the global maximum for this sample.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_max[s_index] = fmaxf(s_max[s_index], s_max[s_index + stride]);
        }
        __syncthreads();
    }
    float global_max = s_max[sample_in_block * blockDim.x];

    // Next, compute the sum of exp(logits - global_max) in a similar distributed fashion.
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        local_sum += expf(sample_logits[j] - global_max);
    }
    s_sum[s_index] = local_sum;
    __syncthreads();

    // Reduce within the block to get the total sum.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_sum[s_index] += s_sum[s_index + stride];
        }
        __syncthreads();
    }
    float total_sum = s_sum[sample_in_block * blockDim.x];

    // Finally, one thread per sample (threadIdx.x == 0) computes the loss using the target logit.
    if (threadIdx.x == 0) {
        int target_class = targets[sample_index];
        float target_logit = sample_logits[target_class];
        losses[sample_index] = -(target_logit - global_max - logf(total_sum));
    }
}


// Forward function wrapping the kernel call.
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    auto losses = torch::empty({batch_size}, predictions.options());

    // Configure a 2D block layout: 
    // - Use blockDim.x threads to reduce over classes (e.g., 128 threads).
    // - Use blockDim.y to process multiple samples concurrently (e.g., 4 samples per block).
    dim3 block(128, 4);
    int num_blocks = (batch_size + block.y - 1) / block.y;
    size_t shared_mem = 2 * block.x * block.y * sizeof(float); // shared memory for max and sum arrays

    balanced_cross_entropy_loss_kernel<<<num_blocks, block, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return losses.mean();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced CrossEntropyLoss forward (CUDA)");
}
