#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void vectorized_kl_div_kernel(
    const scalar_t* __restrict__ log_predictions,
    const scalar_t* __restrict__ targets,
    scalar_t* __restrict__ output,
    int n) {

    constexpr int vec_size = sizeof(float4) / sizeof(scalar_t);
    const int tid = blockIdx.x * blockDim.x * vec_size + threadIdx.x; 
    const int warp_id = threadIdx.x / 32;
    constexpr int num_warps = 256 / 32;
    __shared__ scalar_t warp_sums[num_warps];

    scalar_t thread_sum = 0;
    
    // Vectorized processing
    if (tid * vec_size < n) {
        float4 logp_vec = reinterpret_cast<const float4*>(log_predictions + tid * vec_size)[0];
        float4 tgt_vec = reinterpret_cast<const float4*>(targets + tid * vec_size)[0];
        
        #pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            scalar_t log_p = reinterpret_cast<scalar_t*>(&logp_vec)[i];
            scalar_t t = reinterpret_cast<scalar_t*>(&tgt_vec)[i];
            thread_sum += expf(log_p) - t * log_p;
        }
    }

    // Non-vectorized tail handling
    for (int i = tid + blockDim.x * gridDim.x; i < n; i += blockDim.x * gridDim.x) {
        scalar_t log_p = log_predictions[i];
        scalar_t t = targets[i];
        thread_sum += expf(log_p) - t * log_p;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // Block reduction
    if (warp_id == 0) {
        scalar_t block_sum = (threadIdx.x < num_warps) ? warp_sums[threadIdx.x] : 0;
        for (int offset = 8; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        
        if (threadIdx.x == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    constexpr int threads = 256;
    const int vec_size = sizeof(float4) / sizeof(float);
    const int blocks = (n + threads * vec_size - 1) / (threads * vec_size);
    
    AT_DISPATCH_FLOATING_TYPES(log_predictions.scalar_type(), "kl_div_forward", ([&] {
        vectorized_kl_div_kernel<scalar_t>
            <<<blocks, threads, (256/32)*sizeof(scalar_t)>>>(
                log_predictions.data_ptr<scalar_t>(),
                targets.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n
            );
    }));
    
    return output / static_cast<scalar_t>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Vectorized KL Div forward (CUDA)");
}
