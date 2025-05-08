#include <torch/extension.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void cross_entropy_coalesced_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    cg::thread_block block = cg::this_thread_block();
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];

    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;

    const float* sample_logits = logits + sample_idx * num_classes;
    int target_class = targets[sample_idx];

    // Coalesced memory read with thread-stride access
    float val = 0.0f, max_val = -INFINITY, sum_exp = 0.0f;
    for (int class_idx = threadIdx.x; class_idx < num_classes; class_idx += blockDim.x) {
        val = sample_logits[class_idx];
        max_val = fmaxf(max_val, val);
    }

    // Warp-level reduction for max
    float warp_max = cg::reduce(block.partitioned<32>(),
                                max_val,
                                cg::greater<float>());

    // Block-level reduction for max
    if (block.thread_rank() < 32) {
        shared_max[block.thread_rank()] = warp_max;
    }
    block.sync();

    if (block.thread_rank() == 0) {
        max_val = shared_max[0];
        for (int i = 1; i < 32; i++) {
            max_val = fmaxf(max_val, shared_max[i]);
        }
        shared_max[0] = max_val;
    }
    block.sync();
    max_val = shared_max[0];

    // Calculate sum_exp with optimized coalesced access
    for (int class_idx = threadIdx.x; class_idx < num_classes; class_idx += blockDim.x) {
        sum_exp += expf(sample_logits[class_idx] - max_val);
    }

    // Warp-level reduction for sum
    float warp_sum = cg::reduce(block.partitioned<32>(),
                                sum_exp,
                                cg::plus<float>());

    // Block-level reduction for sum
    if (block.thread_rank() < 32) {
        shared_sum[block.thread_rank()] = warp_sum;
    }
    block.sync();

    if (block.thread_rank() == 0) {
        sum_exp = shared_sum[0];
        for (int i = 1; i < 32; i++) {
            sum_exp += shared_sum[i];
        }
        float log_sum_exp = logf(sum_exp);

        // Final calculation by first thread
        float target_val = sample_logits[target_class];
        losses[sample_idx] = -(target_val - max_val - log_sum_exp);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    auto losses = torch::empty({batch_size}, predictions.options());

    // One block per sample, optimized for coalesced access
    int threads = 256;
    cross_entropy_coalesced_kernel<<<batch_size, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced CE Loss forward");
}
