#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cooperative_groups.h>

using namespace cooperative_groups;

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
    extern __shared__ scalar_t shared_data[];
    
    const int batch = blockIdx.z / channels;
    const int channel = blockIdx.z % channels;
    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x_out >= output_width || y_out >= output_height) return;
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int x_in = x_out * stride - padding;
    const int y_in = y_out * stride - padding;
    
    // Load data into shared memory
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int x = x_in + kw * dilation;
            const int y = y_in + kh * dilation;
            if (x >= 0 && x < input_width && y >= 0 && y < input_height) {
                shared_data[threadIdx.y * blockDim.x + threadIdx.x] = 
                    input[batch * channels * input_height * input_width + 
                         channel * input_height * input_width +
                         y * input_width + x];
            }
            __syncthreads();
            
            // Warp-level reduction
            scalar_t val = shared_data[threadIdx.y * blockDim.x + threadIdx.x];
            for (int offset = 16; offset > 0; offset /= 2)
                val = max(val, __shfl_down_sync(0xffffffff, val, offset));
            
            if (threadIdx.x == 0)
                max_val = max(max_val, val);
        }
    }
    
    if (threadIdx.x == 0 && x_out < output_width && y_out < output_height)
        output[batch * channels * output_height * output_width +
              channel * output_height * output_width +
              y_out * output_width + x_out] = max_val;
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

    dim3 threads(32, 4);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads, threads.x * threads.y * sizeof(scalar_t)>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}