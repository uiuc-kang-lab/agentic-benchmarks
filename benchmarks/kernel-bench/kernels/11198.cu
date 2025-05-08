#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* block_results,
    float* final_output,
    int n_elements,
    float inv_n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    for (int i = idx; i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        if (abs_diff < 1.0f) {
            thread_sum += 0.5f * diff * diff;
        } else {
            thread_sum += abs_diff - 0.5f;
        }
    }

    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    shared_sum[tid] = thread_sum;
    __syncthreads();

    // Warp-level reduction first (assumes warp size = 32)
    int lane = tid % 32;
    int wid = tid / 32;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Write reduced warp results to shared memory
    if (lane == 0) {
        shared_sum[wid] = thread_sum;
    }
    __syncthreads();

    // Final reduction across warps
    if (tid < 8) {  // Assuming block size = 256, so 8 warps
        float sum = shared_sum[tid];
        #pragma unroll
        for (int i = 8; i > 0; i >>= 1) {
            sum += __shfl_down_sync(0xff, sum, i);
        }
        if (tid == 0) {
            block_results[blockIdx.x] = sum;
        }
    }

    __syncthreads();

    // Only one thread per grid performs the final reduction
    if (idx == 0) {
        float final_sum = 0.0f;
        for (int i = 0; i < gridDim.x; i++) {
            final_sum += block_results[i];
        }
        *final_output = final_sum * inv_n_elements;
    }
}

torch::Tensor smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(
        predictions.sizes() == targets.sizes(),
        "Input tensors must have the same shape"
    );
    TORCH_CHECK(
        predictions.is_contiguous() && targets.is_contiguous(),
        "Input tensors must be contiguous"
    );
    TORCH_CHECK(
        predictions.device().is_cuda() && targets.device().is_cuda(),
        "Inputs must be CUDA tensors"
    );

    int n = predictions.numel();
    float inv_n_elements = 1.0f / static_cast<float>(n);

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    auto block_results = torch::empty({grid_size}, predictions.options());
    auto output = torch::zeros({1}, predictions.options());

    smooth_l1_loss_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        inv_n_elements
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Smooth L1 Loss (CUDA)");
}