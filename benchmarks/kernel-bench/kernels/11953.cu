#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for vectorized loading of float data
template <typename scalar_t>
__device__ __forceinline__ void load_vector4(
    const scalar_t* __restrict__ src,
    int idx,
    float4& dst) {
    float4* vec_ptr = reinterpret_cast<float4*>(&dst);
    *vec_ptr = *reinterpret_cast<const float4*>(&src[idx]);
}

// Device function for computing squared differences
template <typename scalar_t>
__device__ __forceinline__ void compute_squared_diff(
    const float4& a,
    const float4& b,
    scalar_t& result) {
    float diff0 = a.x - b.x;
    float diff1 = a.y - b.y;
    float diff2 = a.z - b.z;
    float diff3 = a.w - b.w;
    result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
}

// Device function for warp-level reduction
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Device function for block-level reduction
template <typename scalar_t>
__device__ __forceinline__ void block_reduce(
    scalar_t& val,
    scalar_t* shared_mem,
    int tid,
    int lane,
    int warp_id) {
    
    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (tid < (blockDim.x / 32)) ? shared_mem[lane] : 0;
        val = warp_reduce(val);
    }
}

// Device function for computing distances
template <typename scalar_t>
__device__ __forceinline__ void compute_distances(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    int base_idx,
    int feat_size,
    int tid,
    int blockDim,
    scalar_t& dist_pos,
    scalar_t& dist_neg) {
    
    if constexpr (std::is_same<scalar_t, float>::value) {
        float4 a_vec, p_vec, n_vec;
        const int vec_size = 4;
        const int vectorized_length = feat_size / vec_size;
        const int remainder = feat_size % vec_size;
        
        for (int i = tid; i < vectorized_length; i += blockDim) {
            const int vec_idx = base_idx + i * vec_size;
            load_vector4(anchor, vec_idx, a_vec);
            load_vector4(positive, vec_idx, p_vec);
            load_vector4(negative, vec_idx, n_vec);
            
            compute_squared_diff(a_vec, p_vec, dist_pos);
            compute_squared_diff(a_vec, n_vec, dist_neg);
        }
        
        const int offset = vectorized_length * vec_size;
        for (int i = tid; i < remainder; i += blockDim) {
            const int idx = base_idx + offset + i;
            const float a = __ldg(&anchor[idx]);
            const float p = __ldg(&positive[idx]);
            const float n = __ldg(&negative[idx]);
            
            const float diff_p = a - p;
            const float diff_n = a - n;
            dist_pos += diff_p * diff_p;
            dist_neg += diff_n * diff_n;
        }
    }
}

template <typename scalar_t>
__global__ void modular_aligned_triplet_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int base_idx = batch_idx * feat_size;
    
    scalar_t local_dist_pos = 0;
    scalar_t local_dist_neg = 0;
    
    // Compute distances using vectorized loads
    compute_distances(
        anchor, positive, negative,
        base_idx, feat_size, tid, blockDim.x,
        local_dist_pos, local_dist_neg);
    
    // Warp-level reduction
    local_dist_pos = warp_reduce(local_dist_pos);
    local_dist_neg = warp_reduce(local_dist_neg);
    
    __shared__ scalar_t shared_mem[64]; // Space for both pos and neg reductions
    
    // Block-level reduction for positive distances
    block_reduce(
        local_dist_pos,
        shared_mem,
        tid, lane, warp_id);
    
    // Block-level reduction for negative distances
    block_reduce(
        local_dist_neg,
        &shared_mem[32],
        tid, lane, warp_id);
    
    if (warp_id == 0 && lane == 0) {
        const scalar_t loss = sqrt(local_dist_pos) - sqrt(local_dist_neg) + margin;
        output[batch_idx] = loss < scalar_t(0) ? scalar_t(0) : loss;
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
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "modular_aligned_triplet_kernel", ([&] {
        modular_aligned_triplet_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
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