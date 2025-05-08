#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void pipelined_triplet_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size,
    const int batch_offset) {

    int batch_idx = blockIdx.x + batch_offset;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;
    int base_idx = batch_idx * feat_size;
    scalar_t local_dist_pos = 0;
    scalar_t local_dist_neg = 0;

    // Vectorized loads using float4/double2
    if constexpr (std::is_same<scalar_t, float>::value) {
        using vec_t = float4;
        const int vec_size = 4;
        const int vectorized_length = feat_size / vec_size;
        const int remainder = feat_size % vec_size;
        
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

        const int offset = vectorized_length * vec_size;
        #pragma unroll 4
        for (int i = tid; i < remainder; i += blockDim.x) {
            const int idx = base_idx + offset + i;
            const float a = __ldg(&anchor[idx]);
            const float p = __ldg(&positive[idx]);
            const float n = __ldg(&negative[idx]);
            
            const float diff_pos = a - p;
            const float diff_neg = a - n;
            
            local_dist_pos += diff_pos * diff_pos;
            local_dist_neg += diff_neg * diff_neg;
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        local_dist_pos += __shfl_down_sync(0xffffffff, local_dist_pos, offset);
        local_dist_neg += __shfl_down_sync(0xffffffff, local_dist_neg, offset);
    }

    const int lane = tid % warpSize;
    const int warp_id = tid / warpSize;

    __shared__ scalar_t s_pos[32];
    __shared__ scalar_t s_neg[32];

    if (lane == 0) {
        s_pos[warp_id] = local_dist_pos;
        s_neg[warp_id] = local_dist_neg;
    }
    __syncthreads();

    if (warp_id == 0 && lane < (blockDim.x + 31)/32) {
        local_dist_pos = s_pos[lane];
        local_dist_neg = s_neg[lane];

        #pragma unroll
        for (int offset = (blockDim.x + 31)/64; offset > 0; offset >>= 1) {
            local_dist_pos += __shfl_down_sync(0xffffffff, local_dist_pos, offset);
            local_dist_neg += __shfl_down_sync(0xffffffff, local_dist_neg, offset);
        }

        if (lane == 0) {
            output[batch_idx] = max(scalar_t(0), 
                sqrt(local_dist_pos) - sqrt(local_dist_neg) + margin);
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
    const int num_streams = 4;  // Number of concurrent streams
    const int chunk_size = (batch_size + num_streams - 1) / num_streams;
    
    auto output = torch::zeros({batch_size}, anchor.options());
    
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads_per_block = 256;
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "pipelined_triplet_kernel", ([&] {
        for (int i = 0; i < num_streams; i++) {
            const int stream_offset = i * chunk_size;
            const int stream_size = min(chunk_size, batch_size - stream_offset);
            if (stream_size <= 0) continue;

            const int num_blocks = stream_size;
            
            pipelined_triplet_kernel<scalar_t><<<num_blocks, threads_per_block, 0, streams[i]>>>(
                anchor.data_ptr<scalar_t>(),
                positive.data_ptr<scalar_t>(),
                negative.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                margin,
                batch_size,
                feat_size,
                stream_offset);
        }
    }));

    // Synchronize all streams before computing mean
    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cuda, "Triplet margin loss forward (CUDA)");
}