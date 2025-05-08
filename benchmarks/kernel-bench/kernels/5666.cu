#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

constexpr int TILE_WIDTH = 16;
constexpr int TILE_HEIGHT = 16;

template <typename scalar_t>
__global__ void maxpool2d_coalesced_kernel(
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
    const int dilation)
{
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_mem);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int output_x = blockIdx.x * TILE_WIDTH + tx;
    const int output_y = blockIdx.y * TILE_HEIGHT + ty;
    const int batch = blockIdx.z / channels;
    const int channel = blockIdx.z % channels;
    
    if (batch >= batch_size || channel >= channels) return;
    
    const int input_base = batch * channels * input_height * input_width 
        + channel * input_height * input_width;
    
    const int input_start_x = max(0, output_x * stride - padding);
    const int input_start_y = max(0, output_y * stride - padding);
    const int input_end_x = min(input_width, (output_x + kernel_size) * stride - padding + (dilation - 1) * (kernel_size - 1) + 1);
    const int input_end_y = min(input_height, (output_y + kernel_size) * stride - padding + (dilation - 1) * (kernel_size - 1) + 1);
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    for (int ix = input_start_x; ix < input_end_x; ix += stride) {
        for (int iy = input_start_y; iy < input_end_y; iy += stride) {
            const int input_idx = input_base + iy * input_width + ix;
            shared_input[ty * TILE_WIDTH + tx] = input[input_idx];
            __syncthreads();
            
            #pragma unroll
            for (int kh = 0; kh < kernel_size; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int valid_h = iy + kh * dilation;
                    const int valid_w = ix + kw * dilation;
                    if (valid_h >= 0 && valid_h < input_height && valid_w >= 0 && valid_w < input_width) {
                        scalar_t val = shared_input[ty * TILE_WIDTH + tx];
                        max_val = max(max_val, val);
                    }
                }
            }
        }
    }
    
    if (output_x < output_width && output_y < output_height) {
        output[batch * channels * output_height * output_width + channel * output_height * output_width + output_y * output_width + output_x] = max_val;
    }
}

torch::Tensor maxpool2d_coalesced_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation)
{
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    dim3 threads(TILE_WIDTH, TILE_HEIGHT);
    dim3 blocks(
        (output_width + TILE_WIDTH - 1) / TILE_WIDTH,
        (output_height + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch_size * channels
    );
    
    const size_t shared_size = TILE_WIDTH * TILE_HEIGHT * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool2d_coalesced_forward", ([&] {
        maxpool2d_coalesced_kernel<scalar_t><<<blocks, threads, shared_size>>>(
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

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &maxpool2d_coalesced_forward, "Memory-coalesced Max Pool 2D forward (CUDA)");
}
