#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function for warp-level reduction using shuffle
__inline__ __device__ float warp_reduce_sum(float val) {
    // Assuming warp size of 32
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel that computes hinge loss and reduces the sum using shared memory
__global__ void hinge_loss_reduction_kernel(const float* __restrict__ predictions,
                                             const float* __restrict__ targets,
                                             const int n,
                                             float* __restrict__ sum_out) {
    float local_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple elements in a strided loop
    for (int i = idx; i < n; i += stride) {
        float p = __ldg(&predictions[i]);
        float t = __ldg(&targets[i]);
        float loss = 1.0f - p * t;
        // Only positive losses contribute
        local_sum += (loss > 0.0f ? loss : 0.0f);
    }

    // Warp-level reduction using shuffle instructions
    local_sum = warp_reduce_sum(local_sum);

    // Use shared memory to accumulate results from each warp
    __shared__ float shared_data[32];  // Enough for up to 1024 threads per block
    int lane = threadIdx.x & 31;  // lane index (0-31)
    int warpId = threadIdx.x >> 5; // warp id within block

    // Only the first thread in each warp writes its reduced sum to shared memory
    if (lane == 0) {
        shared_data[warpId] = local_sum;
    }
    // Ensure all warp sums are written before final reduction
    __syncthreads();

    // Let the first warp aggregate the results from all warps in the block
    if (threadIdx.x < (blockDim.x >> 5)) {
        local_sum = shared_data[lane];
        local_sum = warp_reduce_sum(local_sum);
    }

    // The first thread of the block adds the block's total to the global accumulator
    if (threadIdx.x == 0) {
        atomicAdd(sum_out, local_sum);
    }
}

// The forward function allocates an accumulator tensor, launches the kernel, and computes the mean
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    // Allocate a one-element tensor for the sum and initialize to zero
    torch::Tensor sum_tensor = torch::zeros({1}, predictions.options());

    const int threads = 256;
    const int max_blocks = 65535;
    const int blocks = min((n + threads - 1) / threads, max_blocks);

    hinge_loss_reduction_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        n,
        sum_tensor.data_ptr<float>()
    );

    // Compute the mean loss by dividing the accumulated sum by the total number of elements
    return sum_tensor / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Reduced Hinge Loss Forward");
}
