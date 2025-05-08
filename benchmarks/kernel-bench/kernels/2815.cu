#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int THREADS = 256;
const int ELEMENTS_PER_THREAD = 4;

// Kernel that utilizes shared memory to minimize global memory accesses

__global__ void sigmoid_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             const int64_t size) {
    __shared__ float shared_data[THREADS * ELEMENTS_PER_THREAD];
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD;
    const int start_idx = block_offset + tid * ELEMENTS_PER_THREAD;

    // Load input into shared memory
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = start_idx + i;
        if (idx < size) {
            shared_data[tid * ELEMENTS_PER_THREAD + i] = input[idx];
        }
    }
    __syncthreads();

    // Calculate sigmoid and store result in shared memory
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = tid * ELEMENTS_PER_THREAD + i;
        if (start_idx + i < size) {
            float val = shared_data[idx];
            shared_data[idx] = 1.0f / (1.0f + expf(-val));
        }
    }
    __syncthreads();  // Ensure all computations complete

    // Write result from shared memory to output
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = start_idx + i;
        if (idx < size) {
            output[idx] = shared_data[tid * ELEMENTS_PER_THREAD + i];
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    const int blocks = (size + THREADS * ELEMENTS_PER_THREAD - 1) / (THREADS * ELEMENTS_PER_THREAD);

    const auto* input_data = input.data_ptr<float>();
    auto* output_data = output.data_ptr<float>();

    sigmoid_kernel<<<blocks, THREADS>>>(input_data, output_data, size);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Sigmoid forward using shared memory (CUDA)");
}