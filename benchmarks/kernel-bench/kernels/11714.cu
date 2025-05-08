#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce(float val, float* shared) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Warp-level reduction
    val = warp_reduce(val);
    
    // Store warp sums
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces block sums
    if (warp_id == 0) {
        float block_sum = (lane_id < blockDim.x/WARP_SIZE) ? shared[lane_id] : 0.0f;
        block_sum = warp_reduce(block_sum);
        return block_sum;
    }
    return 0.0f;
}

} // anonymous namespace

__global__ void modular_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    extern __shared__ float shared_sums[];
    float sum = 0.0f;
    
    // Grid-stride loop with read-only cache
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += expf(log_pred) - target * log_pred;
    }

    // Hierarchical reduction
    float block_sum = block_reduce(sum, shared_sums);
    
    // Atomic add to global output
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

torch::Tensor modular_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Optimal block size selection
    int block_size = 256;
    if (n > 65536) block_size = 512;
    else if (n < 8192) block_size = 128;

    const int max_blocks = 256;
    int grid_size = min(max_blocks, (n + block_size - 1) / block_size);
    int shared_mem = (block_size / WARP_SIZE) * sizeof(float);

    modular_kl_div_kernel<<<grid_size, block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modular_kl_div_forward, "Modular KLDivLoss with hierarchical reductions");
}