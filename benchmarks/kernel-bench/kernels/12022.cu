#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void hybrid_triplet_margin_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    extern __shared__ char shared_memory[];
    scalar_t* shared_pos = (scalar_t*)shared_memory;
    scalar_t* shared_neg = shared_pos + blockDim.x;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    const int grid_stride = gridDim.x * blockDim.x;
    
    // Initialize accumulators
    scalar_t sum_pos = 0;
    scalar_t sum_neg = 0;
    
    // Grid-stride loop for coalesced memory access
    for (int idx = bid * blockDim.x + tid; idx < batch_size * feat_size; idx += grid_stride) {
        const int b = idx / feat_size;
        const int f = idx % feat_size;
        if (b < batch_size && f < feat_size) {
            const scalar_t a = anchor[idx];
            const scalar_t p = positive[idx];
            const scalar_t n = negative[idx];
            
            const scalar_t dp = a - p;
            const scalar_t dn = a - n;
            sum_pos += dp * dp;
            sum_neg += dn * dn;
        }
    }
    
    // First level reduction: within warp using shuffle
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum_pos += __shfl_down_sync(0xffffffff, sum_pos, offset);
        sum_neg += __shfl_down_sync(0xffffffff, sum_neg, offset);
    }
    
    // Write warp results to shared memory
    if (lane_id == 0) {
        shared_pos[warp_id] = sum_pos;
        shared_neg[warp_id] = sum_neg;
    }
    __syncthreads();
    
    // Second level reduction: across warps using the first warp
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0 && lane_id < num_warps) {
        sum_pos = shared_pos[lane_id];
        sum_neg = shared_neg[lane_id];
        
        // Final warp reduction
        #pragma unroll
        for (int offset = num_warps/2; offset > 0; offset /= 2) {
            sum_pos += __shfl_down_sync(0xffffffff, sum_pos, offset);
            sum_neg += __shfl_down_sync(0xffffffff, sum_neg, offset);
        }
        
        // Write final block result
        if (lane_id == 0) {
            const int batch_idx = bid;
            if (batch_idx < batch_size) {
                const scalar_t dist_pos = sqrt(sum_pos);
                const scalar_t dist_neg = sqrt(sum_neg);
                output[batch_idx] = max(scalar_t(0.0), dist_pos - dist_neg + margin);
            }
        }
    }
}

torch::Tensor hybrid_triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {
    
    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");
    
    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks = min(batch_size, 1024); // Limit max blocks for better occupancy
    
    auto output = torch::zeros({batch_size}, anchor.options());
    
    // Calculate shared memory size
    const int shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "hybrid_triplet_margin_loss_kernel", ([&] {
        hybrid_triplet_margin_loss_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
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
    m.def("forward", &hybrid_triplet_margin_loss_cuda, "Hybrid Triplet Margin Loss forward (CUDA)");
}