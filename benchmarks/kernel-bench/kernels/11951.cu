#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Enhanced kernel with optimized block and thread indexing

// Device function for maximum
template <typename T>
__device__ __forceinline__ T device_max(T a, T b) {
    return a > b ? a : b;
}

// Warp-level reduction using shuffle instructions
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel using optimized block and thread indexing strategy for better parallelism
// Improved mapping of threads to workload
template <typename scalar_t>
__global__ void block_efficiency_triplet_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {

    int batch_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (batch_id >= batch_size) return;

    int thread_id = threadIdx.x;
    int base_idx = batch_id * feat_size + thread_id;
    scalar_t local_dist_pos = 0;
    scalar_t local_dist_neg = 0;

    // Process only the remainder of elements if blockDim.x doesn't perfectly fit feat_size
    for (int i = thread_id; i < feat_size; i += blockDim.x) {
        scalar_t a = __ldg(&anchor[base_idx + i]);
        scalar_t p = __ldg(&positive[base_idx + i]);
        scalar_t n = __ldg(&negative[base_idx + i]);
        scalar_t diff_pos = a - p;
        scalar_t diff_neg = a - n;
        local_dist_pos += diff_pos * diff_pos;
        local_dist_neg += diff_neg * diff_neg;
    }

    // Warp-level reduction
    local_dist_pos = warp_reduce_sum(local_dist_pos);
    local_dist_neg = warp_reduce_sum(local_dist_neg);

    // Ensure coherency within block using shared memory
    static __shared__ scalar_t shared_pos[32];
    static __shared__ scalar_t shared_neg[32];

    int lane = thread_id % 32;
    int warp_id = thread_id / 32;

    if (lane == 0) {
        shared_pos[warp_id] = local_dist_pos;
        shared_neg[warp_id] = local_dist_neg;
    }
    __syncthreads();

    scalar_t sum_pos = (lane < blockDim.x / 32) ? shared_pos[lane] : 0;
    scalar_t sum_neg = (lane < blockDim.x / 32) ? shared_neg[lane] : 0;
    sum_pos = warp_reduce_sum(sum_pos);
    sum_neg = warp_reduce_sum(sum_neg);
    if (lane == 0 && warp_id == 0) {
        scalar_t loss = sqrt(sum_pos) - sqrt(sum_neg) + margin;
        output[batch_id] = device_max(scalar_t(0), loss);
    }
}

// Host function to launch the kernel
torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");

    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);

    auto output = torch::zeros({batch_size}, anchor.options());
    
    // Use 2D grid to separate batch elements
    const int threads_per_block_x = 256;
    const int threads_per_block_y = 1;
    dim3 threads(threads_per_block_x, threads_per_block_y);
    dim3 blocks((batch_size + threads.y - 1) / threads.y);
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "block_efficiency_triplet_kernel", ([&] {
        block_efficiency_triplet_kernel<scalar_t><<<blocks, threads>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            margin,
            batch_size,
            feat_size);
    }));
    
    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cuda, "Triplet margin loss forward (CUDA)");
}
