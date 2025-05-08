#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE/WARP_SIZE)

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel(
    const float4* __restrict__ log_predictions4,
    const float4* __restrict__ targets4,
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n4,
    const int n) {
    
    // Setup shared memory for block-level reduction
    __shared__ float smem[WARPS_PER_BLOCK];
    
    // Warp and block indices
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int total_warps = gridDim.x * WARPS_PER_BLOCK;
    
    float sum = 0.0f;

    // Vectorized processing using float4
    for (int base_idx = global_warp_id; base_idx < n4; base_idx += total_warps) {
        float4 logp4 = __ldg(&log_predictions4[base_idx]);
        float4 targ4 = __ldg(&targets4[base_idx]);
        sum += expf(logp4.x) - targ4.x * logp4.x;
        sum += expf(logp4.y) - targ4.y * logp4.y;
        sum += expf(logp4.z) - targ4.z * logp4.z;
        sum += expf(logp4.w) - targ4.w * logp4.w;
    }

    // Process remaining elements
    const int remaining_start = n4 * 4;
    const int elements_left = n - remaining_start;
    
    for (int idx = global_warp_id * 4 + lane_id; 
         idx < elements_left; 
         idx += total_warps * 4) {
        
        if (remaining_start + idx < n) {
            float logp = __ldg(&log_predictions[remaining_start + idx]);
            float targ = __ldg(&targets[remaining_start + idx]);
            sum += expf(logp) - targ * logp;
        }
    }

    // Warp-level reduction
    sum = warpReduceSum(sum);

    // Store warp sum to shared memory
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces block sums
    if (warp_id == 0) {
        float block_sum = lane_id < WARPS_PER_BLOCK ? smem[lane_id] : 0.0f;
        block_sum = warpReduceSum(block_sum);
        if (lane_id == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    const int n4 = n / 4;
    auto output = torch::zeros({1}, log_predictions.options());

    // Improved block sizing with auto-scaling
    int blocks;
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    int max_blocks = std::min(4 * props.multiProcessorCount, 1024);
    blocks = std::min(max_blocks, (n + (BLOCK_SIZE*4) - 1) / (BLOCK_SIZE*4));

    // Reinterpret for vectorized access
    const float4* log_predictions4 = reinterpret_cast<const float4*>(log_predictions.data_ptr<float>());
    const float4* targets4 = reinterpret_cast<const float4*>(targets.data_ptr<float>());

    kl_div_kernel<<<blocks, BLOCK_SIZE>>>(
        log_predictions4,
        targets4,
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n4,
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}