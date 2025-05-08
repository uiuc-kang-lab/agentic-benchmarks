#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel computes the hinge loss for each element and performs an intra-block reduction
// using shared memory. Each thread processes multiple elements via a grid-stride loop, and
// then the block sum is atomically added to a global accumulator. This eliminates the need
// for a separate reduction kernel and leverages shared memory to reduce global memory latency.
__global__ void hinge_loss_reduce_kernel(const float* __restrict__ predictions, 
                                          const float* __restrict__ targets, 
                                          float* global_sum, 
                                          int n) {
    extern __shared__ float sdata[]; // Shared memory allocation (size: blockDim.x floats)
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    float local_sum = 0.0f;

    // Grid-stride loop: each thread processes multiple elements
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float prod = predictions[i] * targets[i];
        float loss = (prod < 1.0f) ? (1.0f - prod) : 0.0f;
        local_sum += loss;
    }
    
    // Store the local sum into shared memory
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduce the sums in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Atomically add the block's result to the global sum
    if (tid == 0) {
        atomicAdd(global_sum, sdata[0]);
    }
}

// The forward function sets up the kernel and computes the mean hinge loss directly.
// It allocates a tensor for the global sum, launches the kernel using dynamic shared memory,
// and then computes the mean by dividing the accumulated hinge loss by the number of elements.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    // Allocate a tensor for the global sum and initialize to 0
    torch::Tensor global_sum_tensor = torch::zeros({1}, predictions.options());

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Launch the kernel with dynamic shared memory size: threads * sizeof(float)
    hinge_loss_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        global_sum_tensor.data_ptr<float>(),
        n
    );
    
    // Retrieve the global sum and compute the mean hinge loss
    float total_loss = global_sum_tensor.item<float>();
    float mean_loss = total_loss / static_cast<float>(n);
    
    // Return the mean as a 0-dimensional tensor
    return torch::full({}, mean_loss, predictions.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward with Shared Memory Reduction");
}
