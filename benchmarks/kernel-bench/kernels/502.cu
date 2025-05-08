#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Kernel using shared memory and warp-level primitives for reduction
__global__ void sharedMemoryReductionKernel(const float* __restrict__ A,
                                             float* __restrict__ C,
                                             float s,
                                             int64_t size) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    // Perform a grid-stride loop
    for (int i = idx; i < size; i += stride) {
        sum += A[i] * s;
    }

    // Store local sum in shared memory
    sharedData[tid] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        C[blockIdx.x] = sharedData[0];
    }
}

// Function to call the kernel and perform final reduction on CPU
torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Allocate memory for block results
    auto blockResults = torch::empty({blocks}, torch::dtype(torch::kFloat).device(torch::kCUDA));

    // Launch kernel
    sharedMemoryReductionKernel<<<blocks, threads, threads * sizeof(float)>>>(A.data_ptr<float>(),
                                                                              blockResults.data_ptr<float>(),
                                                                              s,
                                                                              size);

    // Copy block results back to CPU and perform final reduction
    auto blockResultsHost = blockResults.cpu();
    float finalSum = blockResultsHost.sum().item<float>();

    // Fill the result tensor with the final sum
    C.fill_(finalSum);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared memory reduction matrix-scalar multiplication kernel");
}