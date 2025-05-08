#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Inline function to compute the KL divergence contribution per element
__device__ inline float compute_kldiv_value(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

// Warp-level reduction using shuffle instructions
__device__ inline float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel that minimizes warp divergence by using uniform control flow
// It processes complete groups (of 4 elements) via vectorized loads and handles tail elements in a separate uniform step
__global__ void kl_div_kernel_uniform_nodivergence(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Compute the number of complete groups of 4 elements
    int vec_count = n / 4;  // number of vectorized iterations

    int tid = threadIdx.x;
    int global_thread_id = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    // Process complete groups uniformly with vectorized loads; no conditional branch inside the loop
    // Each thread computes indices in the vectorized domain
    for (int i = global_thread_id; i < vec_count; i += stride) {
        int base = i * 4;
        // Load 4 floats at once from each input array
        float4 log_vec = reinterpret_cast<const float4*>(log_predictions)[i];
        float4 target_vec = reinterpret_cast<const float4*>(targets)[i];

        sum += compute_kldiv_value(log_vec.x, target_vec.x)
             + compute_kldiv_value(log_vec.y, target_vec.y)
             + compute_kldiv_value(log_vec.z, target_vec.z)
             + compute_kldiv_value(log_vec.w, target_vec.w);
    }

    // Perform warp-level reduction with shuffles to sum values within each warp
    sum = warp_reduce(sum);
    
    // Use shared memory to accumulate warp sums into a block-level sum
    __shared__ float shared[32]; // sufficient for blocks up to 1024 threads (32 warps max)
    int warp_id = tid / 32;

    // First thread in each warp stores its warp reduced value
    if ((tid & 31) == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    // Let the first warp perform a final reduction over the warp sums
    int numWarps = (blockDim.x + 31) / 32;
    if (tid < numWarps) {
        sum = shared[tid];
        sum = warp_reduce(sum);
        if (tid == 0) {
            atomicAdd(output, sum);
        }
    }

    // Process tail elements (n not divisible by 4) in a uniform way
    // Since tail is small (<4 elements), we let only one thread (block 0, thread 0) handle it
    if (blockIdx.x == 0 && tid == 0) {
        float tail_sum = 0.0f;
        int tail_start = vec_count * 4;
        for (int j = tail_start; j < n; j++) {
            tail_sum += compute_kldiv_value(log_predictions[j], targets[j]);
        }
        atomicAdd(output, tail_sum);
    }
}

// Host function that wraps the kernel for a PyTorch CUDA extension
torch::Tensor kl_div_cuda_forward_uniform_nodivergence(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    // Compute number of vectorized iterations; use grid-stride loop over the vector domain
    int vec_count = n / 4;
    const int blocks = min((vec_count + threads - 1) / threads, 1024);

    kl_div_kernel_uniform_nodivergence<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_uniform_nodivergence, "Uniform control flow KLDiv forward (CUDA)");
}
