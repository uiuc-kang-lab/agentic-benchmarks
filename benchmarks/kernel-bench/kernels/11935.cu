#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void triplet_margin_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    // Calculate grid stride for persistent threads
    const int grid_stride = gridDim.x * blockDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = thread_id / 32;
    const int lane_id = thread_id % 32;
    
    // Process multiple samples per thread using grid stride
    for (int sample = warp_id; sample < batch_size; sample += grid_stride / 32) {
        scalar_t dist_pos = 0;
        scalar_t dist_neg = 0;
        
        // Stride through features for this sample
        const int sample_offset = sample * feat_size;
        
        #pragma unroll 4
        for (int feat = lane_id; feat < feat_size; feat += 32) {
            const int idx = sample_offset + feat;
            const scalar_t a = anchor[idx];
            const scalar_t p = positive[idx];
            const scalar_t n = negative[idx];
            
            const scalar_t d_pos = a - p;
            const scalar_t d_neg = a - n;
            
            dist_pos += d_pos * d_pos;
            dist_neg += d_neg * d_neg;
        }
        
        // Warp reduction for this sample
        dist_pos = warp_reduce_sum(dist_pos);
        dist_neg = warp_reduce_sum(dist_neg);
        
        // First thread in warp writes result
        if (lane_id == 0) {
            const scalar_t loss = max(scalar_t(0.0), sqrt(dist_pos) - sqrt(dist_neg) + margin);
            output[sample] = loss;
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
    
    // Calculate optimal grid size based on SM count
    int max_blocks_per_sm;
    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, 
        triplet_margin_loss_kernel<float>, 256, 0);
    
    const int thread_count = 256;
    const int max_blocks = max_blocks_per_sm * num_sm;
    const int num_blocks = min(max_blocks, (batch_size * 32 + thread_count - 1) / thread_count);
    
    auto output = torch::zeros({batch_size}, anchor.options());
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel", ([&] {
        triplet_margin_loss_kernel<scalar_t><<<num_blocks, thread_count>>>(
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