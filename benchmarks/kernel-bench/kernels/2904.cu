#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using shared memory and warp-level primitives for optimized reductions

template <typename scalar_t>
__global__ void tanh_kernel_shared(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    extern __shared__ scalar_t shared_data[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    if (idx < size) {
        // Compute tanh activation for the element
        scalar_t x = input[idx];
        scalar_t y = tanhf(x);
        output[idx] = y;

        // Store the result in shared memory
        shared_data[tid] = y;
        __syncthreads();

        // Perform block-level reduction using shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                shared_data[tid] += shared_data[tid + stride];
            }
            __syncthreads();
        }

        // Use warp-level primitives for the final reduction stage
        if (tid < warpSize) {
            scalar_t warp_sum = shared_data[tid];
            unsigned int mask = __ballot_sync(0xffffffff, tid < warpSize);
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(mask, warp_sum, offset);
            }
            // Optionally, store the warp_sum for further use
            if (tid == 0) {
                // This could be used for further computation or logging
            }
        }
    }
}

// Forward function wrapping the kernel launch

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_shared", ([&] {
        tanh_kernel_shared<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with shared memory and warp-level primitives (CUDA)");
}
