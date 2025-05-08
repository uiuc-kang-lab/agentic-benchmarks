#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define shared memory usage for temporary data storage

template <typename scalar_t>
__global__ void softplus_kernel_shared(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, const int size) {
    extern __shared__ scalar_t sdata[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Load input to shared memory
        sdata[threadIdx.x] = input[idx];
        __syncthreads(); // Ensure all loads are completed

        // Perform computation in parallel
        scalar_t x = sdata[threadIdx.x];
        if (x > static_cast<scalar_t>(20.0)) {
            sdata[threadIdx.x] = x;
        } else if (x < static_cast<scalar_t>(-20.0)) {
            sdata[threadIdx.x] = exp(x);
        } else {
            sdata[threadIdx.x] = log1p(exp(x));
        }

        // Ensure all threads have completed writing to shared memory
        __syncthreads(); // Remove if no intra-block dependency exists

        // Store result back to global memory
        output[idx] = sdata[threadIdx.x];
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_shared<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
