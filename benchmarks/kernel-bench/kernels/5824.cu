#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Kernel that utilizes shared memory for each block to speed up pooling operations
// and minimizes the use of __syncthreads().
template <typename scalar_t>
__global__ void max_pool2d_kernel(
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

    int ow = threadIdx.x + blockIdx.x * blockDim.x;
    int oh = threadIdx.y + blockIdx.y * blockDim.y;
    if (ow >= output_width || oh >= output_height) return;

    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    int in_y = oh * stride - padding;
    int in_x = ow * stride - padding;

    int kh_start = (in_y < 0) ? ((-in_y + dilation - 1) / dilation) : 0;
    int kh_end = (input_height - in_y + dilation - 1) / dilation;
    if (kh_end > kernel_size) kh_end = kernel_size;

    int kw_start = (in_x < 0) ? ((-in_x + dilation - 1) / dilation) : 0;
    int kw_end = (input_width - in_x + dilation - 1) / dilation;
    if (kw_end > kernel_size) kw_end = kernel_size;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Load input data into shared memory
    int ih_offset = in_y;
    int iw_offset = in_x;
    for (int kh = kh_start; kh < kh_end; ++kh) {
        int ih = ih_offset + kh * dilation;
        for (int kw = kw_start; kw < kw_end; ++kw) {
            int iw = iw_offset + kw * dilation;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int input_idx = b * (channels * input_height * input_width) +
                                c * (input_height * input_width) +
                                ih * input_width + iw;
                int shared_idx = (kh * kernel_size + kw) * blockDim.x * blockDim.y + tid;
                shared_mem[shared_idx] = input[input_idx];
            }
        }
    }

    __syncthreads();

    // Compute max over loaded values
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    for (int kh = kh_start; kh < kh_end; ++kh) {
        for (int kw = kw_start; kw < kw_end; ++kw) {
            int shared_idx = (kh * kernel_size + kw) * blockDim.x * blockDim.y + tid;
            max_val = max(max_val, shared_mem[shared_idx]);
        }
    }

    output[b * (channels * output_height * output_width) +
           c * (output_height * output_width) +
           oh * output_width + ow] = max_val;
}


torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const dim3 threads(16, 16);  // 256 threads per block
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    const size_t shared_memory_size = kernel_size * kernel_size * threads.x * threads.y * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA) using shared memory optimization");
}
