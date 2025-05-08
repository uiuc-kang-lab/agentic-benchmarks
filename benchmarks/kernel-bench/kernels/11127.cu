#include <torch/extension.h>

__global__ void cross_entropy_loss_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
)
{
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int class_idx = threadIdx.y;
    
    // Shared memory for reductions
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];

    for(; sample_idx < batch_size; sample_idx += blockDim.x * gridDim.x) {
        const float* sample_logits = logits + sample_idx * num_classes;
        int64_t target = targets[sample_idx];

        // Parallel max reduction
        float thread_max = -INFINITY;
        for(int j = class_idx; j < num_classes; j += blockDim.y) {
            thread_max = fmaxf(thread_max, sample_logits[j]);
        }
        
        // Warp-level reduction
        for(int offset = 16; offset > 0; offset /= 2)
            thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));

        if(class_idx == 0) {
            shared_max[threadIdx.x % 32] = thread_max;
        }
        __syncthreads();

        // Block-level reduction
        float max_val = shared_max[threadIdx.x % 32];
        
        // Parallel sum reduction
        float thread_sum = 0.0f;
        for(int j = class_idx; j < num_classes; j += blockDim.y) {
            thread_sum += expf(sample_logits[j] - max_val);
        }

        // Warp-level reduction
        for(int offset = 16; offset > 0; offset /= 2)
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

        if(class_idx == 0) {
            shared_sum[threadIdx.x % 32] = thread_sum;
        }
        __syncthreads();

        // Final calculation
        if(class_idx == 0) {
            float log_sum_exp = logf(shared_sum[0]);
            losses[sample_idx] = -(sample_logits[target] - max_val - log_sum_exp);
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    
    auto losses = torch::empty({batch_size}, predictions.options());

    // 2D block: x=32 samples, y=32 classes
    dim3 threads(32, 32);
    int blocks = (batch_size + threads.x - 1) / threads.x;
    
    cross_entropy_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel error: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 2D CrossEntropyLoss");
}
