#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// Kernel to perform softmax on each row, ensuring coalesced global memory accesses
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Each block handles one row of the input tensor
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Partition the row among warps so that each warp processes a contiguous segment
    int elementsPerWarp = (num_features + num_warps - 1) / num_warps; // ceiling division
    int start = warp_id * elementsPerWarp;
    int end = start + elementsPerWarp;
    if(end > num_features) end = num_features;

    // Compute local maximum over the assigned segment with coalesced reads
    float local_max = -INFINITY;
    for (int i = start + lane; i < end; i += WARP_SIZE) {
        float val = x[batch_idx * num_features + i];
        local_max = max(local_max, val);
    }

    // Warp-level reduction for max using shuffle operations
    uint32_t mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(mask, local_max, offset));
    }

    // Allocate shared memory: first part for max values, second for sum values
    extern __shared__ float smem[];  // Total size: 2 * num_warps floats
    float* smax = smem;              // size: num_warps
    float* ssum = smem + num_warps;    // size: num_warps

    if (lane == 0) {
        smax[warp_id] = local_max;
    }
    __syncthreads();

    // Reduce max values from all warps in thread 0
    if (tid == 0) {
        float global_max = smax[0];
        for (int i = 1; i < num_warps; i++) {
            global_max = max(global_max, smax[i]);
        }
        smax[0] = global_max;
    }
    __syncthreads();
    float global_max = smax[0];

    // Compute exponentials and accumulate local sum in each warp
    float local_sum = 0.0f;
    for (int i = start + lane; i < end; i += WARP_SIZE) {
        float val = x[batch_idx * num_features + i];
        float exp_val = __expf(val - global_max);
        y[batch_idx * num_features + i] = exp_val; // Temporarily store exp value
        local_sum += exp_val;
    }

    // Warp-level reduction for sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    if (lane == 0) {
        ssum[warp_id] = local_sum;
    }
    __syncthreads();

    // Reduce sums from all warps using thread 0
    if (tid == 0) {
        float global_sum = ssum[0];
        for (int i = 1; i < num_warps; i++) {
            global_sum += ssum[i];
        }
        ssum[0] = global_sum;
    }
    __syncthreads();
    float global_sum = ssum[0];

    // Normalize the exponentials to obtain softmax outputs; accesses are coalesced within each warp's segment
    for (int i = start + lane; i < end; i += WARP_SIZE) {
        int idx = batch_idx * num_features + i;
        y[idx] = y[idx] / global_sum;
    }
}

// Host function to launch the softmax kernel
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid_dim(batch_size);
    dim3 block_dim(THREADS_PER_BLOCK);
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    int shared_mem_size = sizeof(float) * 2 * num_warps; // for smax and ssum
    softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
        return;
    }
}

// PyTorch binding: forward function
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);

    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features);
    return y;
}

// pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}
