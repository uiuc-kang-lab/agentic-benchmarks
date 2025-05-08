#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int ELEMENTS_PER_ITER = 8;

__global__ void efficient_warp_reduce_kl_kernel(
    const float* __restrict__ log_preds,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    
    extern __shared__ float warp_sums[];
    float sum = 0.0f;

    // Coalesced grid-strided loop with register reuse
    for (int idx_base = tid*ELEMENTS_PER_ITER; 
         idx_base < n; 
         idx_base += total_threads*ELEMENTS_PER_ITER)
    {
        #pragma unroll
        for (int i=0; i<ELEMENTS_PER_ITER; ++i) {
            const int idx = idx_base + i;
            if (idx < n) {
                const float lp = __ldg(log_preds + idx);
                const float t = __ldg(targets + idx);
                sum += expf(lp) - t * lp;
            }
        }
    }

    // 1. Intra-warp reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // 2. Store warp sums in shared memory
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (threadIdx.x % WARP_SIZE == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    // 3. Single thread reduces shared memory and atomically adds to global mem
    if (threadIdx.x == 0) {
        float block_total = 0;
        const int warps_per_block = blockDim.x / WARP_SIZE;
        for (int i = 0; i < warps_per_block; ++i)
            block_total += warp_sums[i];
        atomicAdd(output, block_total);
    }
}

torch::Tensor efficient_warp_reduce_kl_forward(
    torch::Tensor log_preds,
    torch::Tensor targets)
{
    const int n = log_preds.numel();
    auto output = torch::zeros({1}, log_preds.options());

    const int threads = 256;
    const int max_blocks = 256;
    const int blocks = min(max_blocks, (n + threads*ELEMENTS_PER_ITER-1)/(threads*ELEMENTS_PER_ITER));
    const int shared_mem = (threads / WARP_SIZE) * sizeof(float);

    efficient_warp_reduce_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_preds.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &efficient_warp_reduce_kl_forward, "KL Divergence with efficient warp reductions (CUDA)");
}
