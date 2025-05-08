#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_hybrid_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding,
    const int dilation
) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* shared_tile = reinterpret_cast<scalar_t*>(shared_mem);

    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate shared memory tile dimensions
    const int sm_w = blockDim.x * stride + (KERNEL_SIZE - 1) * dilation;
    const int sm_h = blockDim.y * stride + (KERNEL_SIZE - 1) * dilation;
    const int in_tile_x = blockIdx.x * blockDim.x * stride - padding;
    const int in_tile_y = blockIdx.y * blockDim.y * stride - padding;

    // Cooperative loading with template-based unrolling
    const int tile_size = sm_w * sm_h;
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < tile_size; i += blockDim.x * blockDim.y) {
        int ty = i / sm_w;
        int tx = i % sm_w;
        int in_x = in_tile_x + tx;
        int in_y = in_tile_y + ty;
        
        shared_tile[i] = (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height)
            ? input[((b * channels + c) * input_height + in_y) * input_width + in_x]
            : -std::numeric_limits<scalar_t>::infinity();
    }
    __syncthreads();

    if (out_x >= output_width || out_y >= output_height) return;

    // Template-unrolled max reduction
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int tile_x = threadIdx.x * stride;
    const int tile_y = threadIdx.y * stride;

    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
        #pragma unroll
        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
            int sx = tile_x + kw * dilation;
            int sy = tile_y + kh * dilation;
            scalar_t val = shared_tile[sy * sm_w + sx];
            max_val = fmaxf(max_val, val);
        }
    }

    output[((b * channels + c) * output_height + out_y) * output_width + out_x] = max_val;
}

torch::Tensor max_pool2d_hybrid_forward(
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

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Adaptive block sizing based on kernel size
    dim3 block(16, 16);
    if (kernel_size >= 5) block = dim3(8, 8);
    
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    // Calculate shared memory requirements
    const int sm_w = block.x * stride + (kernel_size - 1) * dilation;
    const int sm_h = block.y * stride + (kernel_size - 1) * dilation;
    const size_t shared_mem = sm_w * sm_h * input.element_size();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_hybrid_forward", ([&] {
        switch(kernel_size) {
            case 2:
                max_pool2d_hybrid_kernel<scalar_t, 2><<<grid, block, shared_mem>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels,
                    input_height, input_width,
                    output_height, output_width,
                    stride, padding, dilation);
                break;
            case 3:
                max_pool2d_hybrid_kernel<scalar_t, 3><<<grid, block, shared_mem>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels,
                    input_height, input_width,
                    output_height, output_width,
                    stride, padding, dilation);
                break;
            default:
                max_pool2d_hybrid_kernel<scalar_t, -1><<<grid, block, shared_mem>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels,
                    input_height, input_width,
                    output_height, output_width,
                    stride, padding, dilation);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_hybrid_forward, "Hybrid Max Pool 2D with shared mem & template unrolling");
}
