#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses vectorized loads (float4) to ensure memory coalescing
// and processes remaining elements separately.

__global__ void vectorized_coalesced_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Calculate how many groups of 4 elements we can process
    int vectorized_n = n / 4;
    int tail_start = vectorized_n * 4;

    const int warp_size = 32;
    const int lane = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;

    extern __shared__ float shared_warp_sums[];

    float thread_sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process the bulk using vectorized loads: each float4 load brings 4 consecutive floats
    const float4* log_preds_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targets_vec   = reinterpret_cast<const float4*>(targets);

    // Loop over groups of 4 elements
    for (int i = tid; i < vectorized_n; i += stride) {
        float4 lp = __ldg(&log_preds_vec[i]);
        float4 tgt = __ldg(&targets_vec[i]);
        thread_sum += expf(lp.x) - tgt.x * lp.x;
        thread_sum += expf(lp.y) - tgt.y * lp.y;
        thread_sum += expf(lp.z) - tgt.z * lp.z;
        thread_sum += expf(lp.w) - tgt.w * lp.w;
    }

    // Process any remaining elements that don't complete a vector of 4
    for (int i = tail_start + tid; i < n; i += stride) {
        float lp = __ldg(&log_predictions[i]);
        float tgt = __ldg(&targets[i]);
        thread_sum += expf(lp) - tgt * lp;
    }

    // Intra-warp reduction using shuffle to sum thread_sum across the warp
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Write the reduced sum of each warp to shared memory
    if (lane == 0) {
        shared_warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // The first warp aggregates the sums of all warps in the block
    if (warp_id == 0) {
        float block_sum = (lane < warps_per_block) ? shared_warp_sums[lane] : 0.0f;
        for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(output, block_sum);
        }
    }
}


// Host function to launch the kernel with dynamic block sizing and shared memory allocation

torch::Tensor vectorized_coalesced_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Select block size based on problem size
    int block_size = 256;
    if (n > 65536) {
        block_size = 512;
    } else if (n < 8192) {
        block_size = 128;
    }

    int blocks = (n + block_size - 1) / block_size;
    if (blocks > 256) {
        blocks = 256;
    }

    int warps_per_block = block_size / 32;
    int shared_mem = warps_per_block * sizeof(float);

    vectorized_coalesced_kl_div_kernel<<<blocks, block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &vectorized_coalesced_kl_div_forward, "Vectorized and coalesced memory access KLDivLoss (CUDA)");
}
