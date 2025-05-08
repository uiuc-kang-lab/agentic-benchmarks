#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cross_entropy_loss_kernel(const float* logits, const int64_t* targets, float* losses, int batch_size, int num_classes) {
    const float* logits,
    const int64_t* targets,
    float* losses,
    int batch_size,
    int num_classes
)
{
    const int warp_size = 32;
    const int sample_idx = blockIdx.x;
    const int lane_idx = threadIdx.x % warp_size;
    const int num_warps = blockDim.x / warp_size;
    const int warp_id = threadIdx.x / warp_size;

    if (sample_idx >= batch_size) return;

    __shared__ float shared_max[32];  // One per warp
    __shared__ float shared_sum[32];  // One per warp

    // Get pointer to logits for this sample
    const float* sample_logits = logits + sample_idx * num_classes;
    
    // Each thread processes its portion of the logits
    float thread_max = -INFINITY;
    for (int j = lane_idx; j < num_classes; j += warp_size) {
        thread_max = max(thread_max, sample_logits[j]);
    }

    // Reduce max within warp
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    // First thread in warp has the max
    if (lane_idx == 0) {
        shared_max[warp_id] = thread_max;
    }
    __syncthreads();

    float max_val = shared_max[warp_id];

    // Compute sum of exponentials
    float thread_sum = 0.0f;
    for (int j = lane_idx; j < num_classes; j += warp_size) {
        thread_sum += expf(sample_logits[j] - max_val);
    }

    // Reduce sum within warp
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (lane_idx == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // First thread computes final loss
    if (threadIdx.x == 0) {
        float sum_exp = shared_sum[0];
        float log_sum_exp = logf(sum_exp);
        int target = targets[sample_idx];
        losses[sample_idx] = -(sample_logits[target] - max_val - log_sum_exp);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets)
{
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    auto losses = torch::empty({batch_size}, predictions.options());

    // Use one block per sample, with multiple warps per block
    const int threads_per_block = 128;  // 4 warps per block
    cross_entropy_loss_kernel<<<batch_size, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward (CUDA)");
}