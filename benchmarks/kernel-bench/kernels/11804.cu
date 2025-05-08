#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// CUDA kernel for KL divergence computation using vectorized loads for memory coalescing
// This kernel processes data in float4 chunks when possible so that threads in a warp read consecutive memory locations.
__global__ void kl_div_kernel_vectorized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* output,
    const int n) {

    int tid = threadIdx.x;
    int global_thread = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float thread_sum = 0.0f;

    // Process the bulk of the data using vectorized loads (float4) for coalescing
    int n_vec = n / 4; // number of float4 elements
    const float4* log_preds_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targets_vec   = reinterpret_cast<const float4*>(targets);

    // Each thread processes vectorized elements using a grid-stride loop
    for (int i = global_thread; i < n_vec; i += stride) {
        float4 lp = log_preds_vec[i];
        float4 tt = targets_vec[i];
        thread_sum += expf(lp.x) - tt.x * lp.x;
        thread_sum += expf(lp.y) - tt.y * lp.y;
        thread_sum += expf(lp.z) - tt.z * lp.z;
        thread_sum += expf(lp.w) - tt.w * lp.w;
    }

    // Process any remaining elements that don't fit into a float4
    int tail_start = n_vec * 4;
    for (int i = tail_start + global_thread; i < n; i += stride) {
        float lp = log_predictions[i];
        float tt = targets[i];
        thread_sum += expf(lp) - tt * lp;
    }

    // Reduction in shared memory
    extern __shared__ float sdata[];
    sdata[tid] = thread_sum;
    __syncthreads();

    // In-block parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // First thread in block accumulates block result
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// CUDA forward function
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = std::min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);

    kl_div_kernel_vectorized<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Optimized KL divergence forward with coalesced memory accesses (CUDA)");
}
