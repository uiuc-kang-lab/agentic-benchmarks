#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Allow tuning different block sizes (e.g., 32, 64, 128, 256, 512) by defining BLOCK_SIZE
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// CUDA kernel using vectorized loads with float4 and shared memory reduction
__global__ void kl_div_kernel_experimental(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float partial_sums[];
    float sum = 0.0f;
    
    // Vectorized processing using float4 for 128-bit alignment
    const int n4 = n / 4;  // Number of vectorized groups
    const float4* logp_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets);

    int grid_stride = blockDim.x * gridDim.x;
    int vec_idx = idx;
    while (vec_idx < n4) {
        float4 lp = __ldg(&logp_vec[vec_idx]);
        float4 tt = __ldg(&targ_vec[vec_idx]);
        sum += expf(lp.x) - tt.x * lp.x;
        sum += expf(lp.y) - tt.y * lp.y;
        sum += expf(lp.z) - tt.z * lp.z;
        sum += expf(lp.w) - tt.w * lp.w;
        vec_idx += grid_stride;
    }

    // Process remaining elements with scalar accesses
    int remaining_start = n4 * 4;
    int scalar_idx = remaining_start + idx;
    while (scalar_idx < n) {
        float lp = __ldg(&log_predictions[scalar_idx]);
        float tt = __ldg(&targets[scalar_idx]);
        sum += expf(lp) - tt * lp;
        scalar_idx += grid_stride;
    }

    // Store partial sum into shared memory
    partial_sums[threadIdx.x] = sum;
    __syncthreads();

    // Intra-block reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Only one thread per block does the atomic add
    if (threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

// Host function to launch the kernel with experimental block size tuning
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Use experimental block sizes. BLOCK_SIZE can be defined as 32,64,128,256,512.
    const int threads = BLOCK_SIZE;
    int n4 = n / 4;  // number of vectorized groups
    int num_blocks = (n4 + threads - 1) / threads;
    // Cap grid size to avoid launching too many blocks when n is small
    num_blocks = min(num_blocks, 1024);

    int shared_mem = threads * sizeof(float);

    kl_div_kernel_experimental<<<num_blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA Experimental Block Size)");
}
