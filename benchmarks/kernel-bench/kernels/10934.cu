#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized block size based on profiling
static const int BLOCK_SIZE = 128;
static const int WARP_SIZE = 32;

template <typename scalar_t>
__global__ void tuned_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double thread_sum = 0.0;
    
    // Calculate thread index and stride
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int global_idx = blockIdx.x * blockDim.x + tid;
    const int grid_stride = gridDim.x * blockDim.x;

    // Process elements with grid stride loop
    // Use vectorized loads when possible for better memory throughput
    if (sizeof(scalar_t) == sizeof(float) && (num_elements % 2 == 0)) {
        const float2* preds2 = reinterpret_cast<const float2*>(preds);
        const float2* tgts2 = reinterpret_cast<const float2*>(tgts);
        
        for (int idx = global_idx; idx < num_elements/2; idx += grid_stride) {
            float2 pred_pair = __ldg(&preds2[idx]);
            float2 tgt_pair = __ldg(&tgts2[idx]);
            
            double diff1 = static_cast<double>(pred_pair.x) - static_cast<double>(tgt_pair.x);
            double diff2 = static_cast<double>(pred_pair.y) - static_cast<double>(tgt_pair.y);
            
            thread_sum += diff1 * diff1 + diff2 * diff2;
        }
    } else {
        for (int idx = global_idx; idx < num_elements; idx += grid_stride) {
            double pred_val = static_cast<double>(__ldg(&preds[idx]));
            double tgt_val = static_cast<double>(__ldg(&tgts[idx]));
            double diff = pred_val - tgt_val;
            thread_sum += diff * diff;
        }
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // First thread in each warp writes to shared memory
    __shared__ double warp_sums[BLOCK_SIZE/WARP_SIZE];
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final reduction across warps
    if (tid < (BLOCK_SIZE/WARP_SIZE)) {
        double warp_sum = warp_sums[tid];
        
        #pragma unroll
        for (int offset = (BLOCK_SIZE/WARP_SIZE)/2; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }

        if (tid == 0) {
            atomicAdd(sum_out, warp_sum);
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Calculate optimal grid size based on smaller block size
    const int num_sms = 108; // H100 has 108 SMs
    const int blocks_per_sm = 2; // Allow multiple blocks per SM
    const int grid_size = std::min(
        (int)((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE),
        num_sms * blocks_per_sm
    );

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "tuned_mse_cuda", ([&] {
        tuned_mse_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tuned Block Size MSE forward (CUDA)");
}