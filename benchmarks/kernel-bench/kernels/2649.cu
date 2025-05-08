#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation with reduction optimization
template <typename scalar_t>
__global__ void relu_kernel_reduction(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    extern __shared__ scalar_t shared_data[];
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int idx = blockIdx.x * blockDim.x * 2 + tid;

    scalar_t sum = 0;

    if (idx < size) {
        sum += input[idx] > 0 ? input[idx] : 0;
        if (idx + block_size < size) {
            sum += input[idx + block_size] > 0 ? input[idx + block_size] : 0;
        }
    }

    shared_data[tid] = sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty({(input.numel() + 511) / 512}, input.options());
    
    const int threads = 256;
    const int blocks = (input.numel() + threads * 2 - 1) / (threads * 2);
    const int shared_memory_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_reduction", ([&] {
        relu_kernel_reduction<scalar_t><<<blocks, threads, shared_memory_size>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    // Final reduction if necessary
    if (blocks > 1) {
        return forward(output);
    } else {
        return output;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward reduction optimized (CUDA)");
}