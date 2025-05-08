#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ inline float compute_element(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kl_div_kernel_vectorized(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int gid = (blockIdx.x * block_size + tid) * 4;
    const int stride = block_size * gridDim.x * 4;

    float sum = 0.0f;

    // Vectorized processing
    for (int idx = gid; idx < n; idx += stride) {
        if (idx + 3 < n) {
            float4 log_preds = load_float4(log_predictions + idx);
            float4 targs = load_float4(targets + idx);
            
            sum += compute_element(log_preds.x, targs.x);
            sum += compute_element(log_preds.y, targs.y);
            sum += compute_element(log_preds.z, targs.z);
            sum += compute_element(log_preds.w, targs.w);
        } else {
            // Handle remaining elements
            for (int k = 0; k < 4 && (idx + k) < n; k++) {
                sum += compute_element(log_predictions[idx + k], targets[idx + k]);
            }
        }
    }

    // Warp-level reduction
    sum = warp_reduce(sum);

    // Block-level reduction
    __shared__ float shared[32];
    if (tid % 32 == 0) {
        shared[tid / 32] = sum;
    }
    __syncthreads();

    if (tid < 32) {
        sum = shared[tid];
        sum = warp_reduce(sum);
        
        if (tid == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward_vectorized(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = min((n + threads * 4 - 1) / (threads * 4), 1024);

    kl_div_kernel_vectorized<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_vectorized, "Vectorized KLDiv forward (CUDA)");
}