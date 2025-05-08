#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define constants for kernel parameters
template <typename scalar_t>
__device__ void atomicMaxFloatScalar(volatile scalar_t* address, scalar_t val) {
    if (*address >= val) return;  // Avoid atomic operation if not necessary
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(max(val, __int_as_float(assumed))));
    } while (assumed != old);
}

template <typename scalar_t>
__global__ void max_pool2d_atomic_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    extern __shared__ scalar_t shared_mem[];
    int tid = threadIdx.x;
    int channel_idx = blockIdx.z * channels + blockIdx.y;

    if (tid < kernel_size * kernel_size) {
        shared_mem[tid] = -std::numeric_limits<scalar_t>::infinity();
    }
    __syncthreads();

    const int ow = blockIdx.x * blockDim.x + tid;
    const int oh = blockIdx.y;
    const int b = blockIdx.z;
    if (ow >= output_width || oh >= output_height || b >= batch_size) return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding + kw * dilation;

            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int input_idx = b * channels * input_height * input_width + channel_idx * input_height * input_width + ih * input_width + iw;
                atomicMaxFloatScalar(&shared_mem[tid], input[input_idx]);
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        scalar_t block_max = -std::numeric_limits<scalar_t>::infinity();
        for (int i = 0; i < blockDim.x; i++)
            block_max = max(block_max, shared_mem[i]);

        int output_idx = b * channels * output_height * output_width + channel_idx * output_height * output_width + oh * output_width + blockIdx.x;
        output[output_idx] = block_max;
    }
}

torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 128;
    const int blocks_x = (output_width + threads - 1) / threads;
    const int blocks_y = output_height;
    const int blocks_z = batch_size;

    const dim3 threadsDim(threads);
    const dim3 blocksDim(blocks_x, blocks_y, blocks_z);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_atomic_kernel<scalar_t><<<blocksDim, threadsDim, kernel_size * kernel_size * sizeof(scalar_t)>>> (
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with atomic operations (CUDA)");
}
