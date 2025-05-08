#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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
    const int tile_width = blockDim.x * stride + (kernel_size - 1) * dilation;
    const int tile_height = blockDim.y * stride + (kernel_size - 1) * dilation;
    extern __shared__ scalar_t shared_input_flat[];
    // Macro to index the 2D shared memory as [row][col]
    #define shared_input(i, j) shared_input_flat[(i) * tile_width + (j)]
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    const int ox = bx + tx;
    const int oy = by + ty;

    // Collaborative loading into shared memory with improved coalescing
    const int tile_start_x = bx * stride - padding;
    const int tile_start_y = by * stride - padding;

    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += blockDim.y) {
        const int load_y = tile_start_y + ty + i;
        #pragma unroll
        for (int j = 0; j < TILE_DIM; j += blockDim.x) {
            const int load_x = tile_start_x + tx + j;
            
            if (load_y >= 0 && load_y < input_height && 
                load_x >= 0 && load_x < input_width) {
                shared_input[ty + i][tx + j] = input[
                    ((b * channels + c) * input_height + load_y) * input_width + load_x];
            } else {
                shared_input[ty + i][tx + j] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }
    __syncthreads();

    if (ox >= output_width || oy >= output_height) return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih_offset = ty * stride + kh * dilation;
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            const int iw_offset = tx * stride + kw * dilation;
            
            if (ih_offset >= 0 && ih_offset < TILE_DIM && 
                iw_offset >= 0 && iw_offset < TILE_DIM) {
                max_val = max(max_val, shared_input[ih_offset][iw_offset]);
            }
        }
    }

    output[((b * channels + c) * output_height + oy) * output_width + ox] = max_val;
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

    const dim3 threads(16, 16);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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