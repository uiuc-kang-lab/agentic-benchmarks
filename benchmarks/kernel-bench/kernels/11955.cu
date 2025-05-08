#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void block_tuned_triplet_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;
    int base_idx = batch_idx * feat_size;
    scalar_t local_dist_pos = 0;
    scalar_t local_dist_neg = 0;

    // Vectorized loads optimized for 128-bit alignment
    if constexpr (std::is_same<scalar_t, float>::value) {
        using vec_t = float4;
        constexpr int vec_size = 4;
        int vectorized_length = feat_size / vec_size;
        int remainder = feat_size % vec_size;
        const vec_t* anchor_vec = reinterpret_cast<const vec_t*>(anchor + base_idx);
        const vec_t* positive_vec = reinterpret_cast<const vec_t*>(positive + base_idx);
        const vec_t* negative_vec = reinterpret_cast<const vec_t*>(negative + base_idx);

        #pragma unroll 4
        for (int i = tid; i < vectorized_length; i += blockDim.x) {
            vec_t a_vec = __ldg(&anchor_vec[i]);
            vec_t p_vec = __ldg(&positive_vec[i]);
            vec_t n_vec = __ldg(&negative_vec[i]);
            
            float diff0 = a_vec.x - p_vec.x;
            float diff1 = a_vec.y - p_vec.y;
            float diff2 = a_vec.z - p_vec.z;
            float diff3 = a_vec.w - p_vec.w;
            local_dist_pos += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
            
            float diff0n = a_vec.x - n_vec.x;
            float diff1n = a_vec.y - n_vec.y;
            float diff2n = a_vec.z - n_vec.z;
            float diff3n = a_vec.w - n_vec.w;
            local_dist_neg += diff0n * diff0n + diff1n * diff1n + diff2n * diff2n + diff3n * diff3n;
        }

        int offset = vectorized_length * vec_size;
        #pragma unroll
        for (int i = tid; i < remainder; i += blockDim.x) {
            int idx = base_idx + offset + i;
            float a = __ldg(&anchor[idx]);
            float p = __ldg(&positive[idx]);
            float n = __ldg(&negative[idx]);
            float diff = a - p;
            local_dist_pos += diff * diff;
            float diffn = a - n;
            local_dist_neg += diffn * diffn;
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < feat_size; i += blockDim.x) {
            int idx = base_idx + i;
            scalar_t a = __ldg(&anchor[idx]);
            scalar_t p = __ldg(&positive[idx]);
            scalar_t n = __ldg(&negative[idx]);
            scalar_t diff = a - p;
            local_dist_pos += diff * diff;
            scalar_t diffn = a - n;
            local_dist_neg += diffn * diffn;
        }
    }

    // Optimized warp reduction using register shuffle
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        local_dist_pos += __shfl_down_sync(0xffffffff, local_dist_pos, offset);
        local_dist_neg += __shfl_down_sync(0xffffffff, local_dist_neg, offset);
    }

    // Shared memory for inter-warp reduction
    __shared__ scalar_t shared_pos[4];  // For 128 threads = 4 warps
    __shared__ scalar_t shared_neg[4];

    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) {
        shared_pos[warp_id] = local_dist_pos;
        shared_neg[warp_id] = local_dist_neg;
    }
    __syncthreads();

    // Final reduction by first warp only
    if (warp_id == 0 && lane < 4) {
        scalar_t warp_sum_pos = shared_pos[lane];
        scalar_t warp_sum_neg = shared_neg[lane];
        
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            warp_sum_pos += __shfl_down_sync(0xffffffff, warp_sum_pos, offset);
            warp_sum_neg += __shfl_down_sync(0xffffffff, warp_sum_neg, offset);
        }

        if (lane == 0) {
            scalar_t loss = sqrt(warp_sum_pos) - sqrt(warp_sum_neg) + margin;
            output[batch_idx] = loss < scalar_t(0) ? scalar_t(0) : loss;
        }
    }
}

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
    
    // Using 128 threads per block for potentially better occupancy
    const int threads_per_block = 128;
    const int num_blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "block_tuned_triplet_kernel", ([&] {
        block_tuned_triplet_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
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