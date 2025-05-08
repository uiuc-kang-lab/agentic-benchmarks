#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel leveraging shared memory to reduce global memory accesses
__global__ void shared_reduction_hinge_loss_kernel(const float* __restrict__ predictions,
                                                    const float* __restrict__ targets,
                                                    float* global_sum,
                                                    int n) {
    // Declare shared memory for block-level reduction
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data from global memory, compute hinge loss and store into shared memory
    float hinge = 0.0f;
    if (idx < n) {
        float pred = __ldg(&predictions[idx]);
        float targ = __ldg(&targets[idx]);
        hinge = fmaxf(0.0f, 1.0f - pred * targ);
    }
    sdata[tid] = hinge;
    __syncthreads();

    // Reduction in shared memory to sum up hinge loss values for the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread of the block contributes the block's sum to the global sum
    if (tid == 0) {
        atomicAdd(global_sum, sdata[0]);
    }
}

// Forward function that sets up kernel execution
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();

    // Allocate a tensor to store the global sum (single float value) on the GPU
    auto global_sum = torch::zeros({1}, predictions.options());

    // Choose a block size and compute the number of blocks
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Launch the kernel with dynamic shared memory allocation
    shared_reduction_hinge_loss_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        global_sum.data_ptr<float>(),
        n
    );

    // Compute the mean hinge loss by dividing the accumulated sum by n
    torch::Tensor mean = global_sum / n;
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared Memory Reduction Hinge Loss Forward Kernel");
}
