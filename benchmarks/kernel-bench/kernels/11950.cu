#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void shared_mem_triplet_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    extern __shared__ char shared_memory[];
    scalar_t* shared_anchor = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* shared_positive = &shared_anchor[blockDim.x];
    scalar_t* shared_negative = &shared_positive[blockDim.x];
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int base_idx = batch_idx * feat_size;
    scalar_t local_dist_pos = 0;
    scalar_t local_dist_neg = 0;

    // Process data in chunks that fit in shared memory
    const int chunk_size = blockDim.x;
    const int num_chunks = (feat_size + chunk_size - 1) / chunk_size;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int chunk_offset = chunk * chunk_size;
        const int current_chunk_size = min(chunk_size, feat_size - chunk_offset);
        
        // Load chunk into shared memory using vectorized loads
        if (tid < current_chunk_size) {
            const int global_idx = base_idx + chunk_offset + tid;
            
            if constexpr (std::is_same<scalar_t, float>::value) {
                if ((tid & 3) == 0 && tid + 3 < current_chunk_size) {
                    // Vector load for aligned addresses
                    float4 a4 = *reinterpret_cast<const float4*>(&anchor[global_idx]);
                    float4 p4 = *reinterpret_cast<const float4*>(&positive[global_idx]);
                    float4 n4 = *reinterpret_cast<const float4*>(&negative[global_idx]);
                    
                    shared_anchor[tid] = a4.x;
                    shared_anchor[tid + 1] = a4.y;
                    shared_anchor[tid + 2] = a4.z;
                    shared_anchor[tid + 3] = a4.w;
                    
                    shared_positive[tid] = p4.x;
                    shared_positive[tid + 1] = p4.y;
                    shared_positive[tid + 2] = p4.z;
                    shared_positive[tid + 3] = p4.w;
                    
                    shared_negative[tid] = n4.x;
                    shared_negative[tid + 1] = n4.y;
                    shared_negative[tid + 2] = n4.z;
                    shared_negative[tid + 3] = n4.w;
                } else if ((tid & 3) != 0) {
                    // Already handled by vector load
                } else {
                    // Handle remaining elements
                    shared_anchor[tid] = __ldg(&anchor[global_idx]);
                    shared_positive[tid] = __ldg(&positive[global_idx]);
                    shared_negative[tid] = __ldg(&negative[global_idx]);
                }
            } else {
                shared_anchor[tid] = __ldg(&anchor[global_idx]);
                shared_positive[tid] = __ldg(&positive[global_idx]);
                shared_negative[tid] = __ldg(&negative[global_idx]);
            }
        }
        __syncthreads();
        
        // Process data from shared memory
        for (int i = tid; i < current_chunk_size; i += blockDim.x) {
            const scalar_t a = shared_anchor[i];
            const scalar_t p = shared_positive[i];
            const scalar_t n = shared_negative[i];
            
            const scalar_t d_pos = a - p;
            const scalar_t d_neg = a - n;
            
            local_dist_pos += d_pos * d_pos;
            local_dist_neg += d_neg * d_neg;
        }
        __syncthreads();
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        local_dist_pos += __shfl_down_sync(0xffffffff, local_dist_pos, offset);
        local_dist_neg += __shfl_down_sync(0xffffffff, local_dist_neg, offset);
    }

    // Block-level reduction using shared memory
    __shared__ scalar_t warp_results_pos[32];
    __shared__ scalar_t warp_results_neg[32];

    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;

    if (lane_id == 0) {
        warp_results_pos[warp_id] = local_dist_pos;
        warp_results_neg[warp_id] = local_dist_neg;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
        local_dist_pos = warp_results_pos[lane_id];
        local_dist_neg = warp_results_neg[lane_id];
        
        #pragma unroll
        for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset >>= 1) {
            local_dist_pos += __shfl_down_sync(0xffffffff, local_dist_pos, offset);
            local_dist_neg += __shfl_down_sync(0xffffffff, local_dist_neg, offset);
        }

        if (lane_id == 0) {
            scalar_t loss = sqrt(local_dist_pos) - sqrt(local_dist_neg) + margin;
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
    
    const int threads_per_block = 256;
    const int num_blocks = batch_size;
    
    // Calculate shared memory size (3 arrays: anchor, positive, negative)
    const size_t shared_mem_size = 3 * threads_per_block * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "shared_mem_triplet_kernel", ([&] {
        shared_mem_triplet_kernel<scalar_t><<<num_blocks, threads_per_block, shared_mem_size>>>(
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