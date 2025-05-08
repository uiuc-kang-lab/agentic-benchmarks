#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute the Smooth L1 (Huber) Loss for a given difference.
__device__ inline float huber_loss(float diff) {
    float abs_diff = fabsf(diff);
    return (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
}

// Device function to perform warp-level reduction using shuffle operations.
__device__ inline float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA kernel that computes the smooth L1 loss using modular device functions
__global__ void smooth_l1_loss_modular_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    float thread_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop over input elements
    for (int i = idx; i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        thread_sum += huber_loss(diff);
    }

    // Reduce within a warp
    thread_sum = warpReduceSum(thread_sum);

    // Allocate shared memory for block-level reduction
    __shared__ float shared_data[32];  // Assuming maximum 32 warps per block
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    
    // Write reduced value of each warp to shared memory
    if (lane == 0) {
        shared_data[warpId] = thread_sum;
    }
    __syncthreads();

    // Only the first warp reduces the per-warp sums
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (warpId == 0) {
        thread_sum = (lane < numWarps) ? shared_data[lane] : 0.0f;
        thread_sum = warpReduceSum(thread_sum);
    }

    // Thread 0 of each block writes the block's contribution to the output (average loss)
    if (threadIdx.x == 0) {
        atomicAdd(output, thread_sum / n_elements);
    }
}

// Host function to set up kernel launch
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
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Calculate optimal grid size based on device multiprocessors
    const int max_blocks_per_sm = 32;  // Typical value for modern GPUs
    const int num_sms = prop.multiProcessorCount;
    const int target_blocks = num_sms * max_blocks_per_sm;
    
    // Ensure we have enough blocks to cover the data while maintaining good occupancy
    const int min_blocks_needed = (n + block_size - 1) / block_size;
    const int grid_size = min(min_blocks_needed, target_blocks);

    smooth_l1_loss_modular_kernel<<<grid_size, block_size>>>(
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
