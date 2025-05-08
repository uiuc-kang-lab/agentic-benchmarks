#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel using grid-stride loops with loop unrolling
// This kernel processes 4 elements per thread iteration, checking boundaries individually.

template <typename scalar_t>
__global__ void grid_stride_unroll_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Loop unrolling factor: 4
    for (int i = idx; i < size; i += stride * 4) {
        int i0 = i;
        int i1 = i + stride;
        int i2 = i + 2 * stride;
        int i3 = i + 3 * stride;

        if (i0 < size) {
            scalar_t val0 = input[i0];
            output[i0] = val0 > static_cast<scalar_t>(0) ? val0 : static_cast<scalar_t>(0);
        }
        if (i1 < size) {
            scalar_t val1 = input[i1];
            output[i1] = val1 > static_cast<scalar_t>(0) ? val1 : static_cast<scalar_t>(0);
        }
        if (i2 < size) {
            scalar_t val2 = input[i2];
            output[i2] = val2 > static_cast<scalar_t>(0) ? val2 : static_cast<scalar_t>(0);
        }
        if (i3 < size) {
            scalar_t val3 = input[i3];
            output[i3] = val3 > static_cast<scalar_t>(0) ? val3 : static_cast<scalar_t>(0);
        }
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    // Compute the number of blocks based on total elements
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_stride_unroll_relu_kernel", ([&] {
        grid_stride_unroll_relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Grid-stride unrolled ReLU forward (CUDA) using stride loops with boundary checks");
}
