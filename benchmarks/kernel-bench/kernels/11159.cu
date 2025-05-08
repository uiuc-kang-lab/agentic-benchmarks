#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    __shared__ float shared_sum[128];  // Reduced shared memory size
    const unsigned int tid = threadIdx.x;
    const unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warpId = tid / warpSize;
    const unsigned int laneId = tid % warpSize;
    const unsigned int grid_stride = gridDim.x * blockDim.x;
    
    float thread_sum = 0.0f;

    // Coalesced global memory access with grid stride loop
    for (int i = gtid; i < n_elements; i += grid_stride) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 
                     (0.5f * diff * diff) : 
                     (abs_diff - 0.5f);
    }

    // First warp-level reduction
    thread_sum = warp_reduce(thread_sum);

    // Write reduced warp results to shared memory
    if (laneId == 0) {
        shared_sum[warpId] = thread_sum;
    }
    __syncthreads();

    // Final reduction using first warp
    if (warpId == 0) {
        thread_sum = (tid < (blockDim.x / warpSize)) ? shared_sum[laneId] : 0.0f;
        thread_sum = warp_reduce(thread_sum);

        if (laneId == 0) {
            atomicAdd(output, thread_sum / n_elements);
        }
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
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    const int grid_size = std::min(256, (n + block_size - 1) / block_size);

    smooth_l1_loss_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Smooth L1 Loss (CUDA)");
}