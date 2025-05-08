#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Kernel Stage 1: Each block computes a partial sum over its assigned elements without using global atomics.
__global__ void kl_div_kernel_stage1(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ block_sums,
    const int n) {

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    // Process using vectorized loads with float4
    int n_vec = n / 4;  // number of groups of 4
    const float4* logp_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets);

    for (int i = global_id; i < n_vec; i += stride) {
        float4 lp = __ldg(&logp_vec[i]);
        float4 tt = __ldg(&targ_vec[i]);
        local_sum += expf(lp.x) - tt.x * lp.x;
        local_sum += expf(lp.y) - tt.y * lp.y;
        local_sum += expf(lp.z) - tt.z * lp.z;
        local_sum += expf(lp.w) - tt.w * lp.w;
    }

    // Process remaining scalar elements
    int scalar_start = n_vec * 4;
    for (int i = scalar_start + global_id; i < n; i += stride) {
        float lp = __ldg(&log_predictions[i]);
        float tt = __ldg(&targets[i]);
        local_sum += expf(lp) - tt * lp;
    }

    // Each thread writes its local sum to shared memory
    sdata[tid] = local_sum;
    __syncthreads();

    // Intra-block reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    // The first thread in the block writes the block's result to global memory
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel Stage 2: Reduce the partial sums from all blocks in a single block without atomics.
__global__ void kl_div_reduce_kernel(
    const float* __restrict__ block_sums,
    float* __restrict__ result,
    const int num_elements) {

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Load partial sums into shared memory
    if (tid < num_elements) {
        sdata[tid] = block_sums[tid];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // Reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset && (tid + offset) < num_elements) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    // Write the final result; only thread 0 does this.
    if (tid == 0) {
        *result = sdata[0];
    }
}

// Host function that launches the two-stage reduction kernel.
// This function computes the KL divergence using two separate kernel launches, avoiding global atomic operations.

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto options = log_predictions.options();

    // Configure Stage 1: vectorized processing; using 256 threads per block.
    const int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;  // based on vectorized (float4) loads
    blocks = std::min(blocks, 1024);

    // Allocate a tensor to hold partial sums (one per block)
    auto block_sums = torch::empty({blocks}, options);

    size_t shared_mem_bytes = threads * sizeof(float);
    kl_div_kernel_stage1<<<blocks, threads, shared_mem_bytes>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        n
    );

    // Stage 2: Reduce the partial block sums to a final sum.
    auto output = torch::empty({1}, options);
    
    // Determine the number of threads for reduction as the next power of 2 of 'blocks'.
    int threads_reduce = 1;
    while (threads_reduce < blocks) { 
        threads_reduce *= 2;
    }

    shared_mem_bytes = threads_reduce * sizeof(float);
    kl_div_reduce_kernel<<<1, threads_reduce, shared_mem_bytes>>>(
        block_sums.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks
    );

    // Return the average KL divergence
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (Two-Stage Reduction, Minimal Atomics)");
}
