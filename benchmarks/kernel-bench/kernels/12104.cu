#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel computes the hinge loss (max(0, 1 - pred*target)) for each element
// and performs an in-block reduction of the loss values. Each block accumulates its partial
// sum into a global accumulator using an atomicAdd.
// We use grid-stride loops with __ldg loads for better memory throughput and __restrict__ qualifiers.
__global__ void fused_hinge_loss_reduction(const float* __restrict__ predictions,
                                           const float* __restrict__ targets,
                                           float* __restrict__ global_sum,
                                           int n) {
    extern __shared__ float sdata[]; // shared memory for block-level reduction
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Grid-stride loop: each thread processes multiple elements
    // #pragma unroll can be added if loop count is known to be small
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float pred = __ldg(&predictions[i]);
        float targ = __ldg(&targets[i]);
        float loss = 1.0f - pred * targ;
        sum += (loss > 0.0f) ? loss : 0.0f;
    }

    sdata[tid] = sum;
    __syncthreads();

    // In-block reduction to sum up values computed by the threads
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread in the block performs an atomic add to global sum
    if (tid == 0) {
        atomicAdd(global_sum, sdata[0]);
    }
}

// The forward function sets up the kernel launch and computes the mean hinge loss.
// It creates an accumulator tensor initialized to zero, launches the fused kernel
// asynchronously on a CUDA stream, synchronizes, and then computes the mean by dividing
// by the total number of elements.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();

    // Create a scalar tensor for the accumulator and initialize to zero
    auto sum_tensor = torch::zeros({1}, predictions.options());

    int threads = 256;
    // Choose the number of blocks to at least cover the workload
    int blocks = (n + threads - 1) / threads;

    // Create a CUDA stream for asynchronous kernel execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Launch the fused kernel with dynamic shared memory of size equal to threads * sizeof(float)
    fused_hinge_loss_reduction<<<blocks, threads, threads * sizeof(float), stream>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
        n
    );

    // Wait for the kernel to complete
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Compute the mean hinge loss by dividing the total sum by number of elements
    auto mean_tensor = sum_tensor / static_cast<float>(n);
    return mean_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused Hinge Loss with Reduction Kernel");
}
