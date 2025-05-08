#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_pool2d_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int TILE_DIM = 16;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(shared_mem);

    const int output_tile_x = blockIdx.x * TILE_DIM;
    const int output_tile_y = blockIdx.y * TILE_DIM;
    const int bc = blockIdx.z;

    // Calculate input tile boundaries
    const int input_start_x = output_tile_x * stride - padding;
    const int input_start_y = output_tile_y * stride - padding;
    const int input_tile_width = TILE_DIM * stride + (kernel_size - 1) * dilation;
    const int input_tile_height = TILE_DIM * stride + (kernel_size - 1) * dilation;

    // Load input tile into shared memory
    for (int ty = threadIdx.y; ty < input_tile_height; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < input_tile_width; tx += blockDim.x) {
            const int y = input_start_y + ty;
            const int x = input_start_x + tx;
            scalar_t val = -std::numeric_limits<scalar_t>::infinity();
            if (y >= 0 && y < input_height && x >= 0 && x < input_width) {
                val = input[bc * input_height * input_width + y * input_width + x];
            }
            shmem[ty * input_tile_width + tx] = val;
        }
    }
    __syncthreads();

    // Compute output elements
    const int y = output_tile_y + threadIdx.y;
    const int x = output_tile_x + threadIdx.x;
    if (y < output_height && x < output_width) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        #pragma unroll
        for (int ky = 0; ky < kernel_size; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int ty = threadIdx.y * stride + ky * dilation;
                const int tx = threadIdx.x * stride + kx * dilation;
                if (ty < input_tile_height && tx < input_tile_width) {
                    max_val = max(max_val, shmem[ty * input_tile_width + tx]);
                }
            }
        }

        output[bc * output_height * output_width + y * output_width + x] = max_val;
    }
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

    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int batch_channels = batch_size * channels;
    const dim3 threads(TILE_DIM, TILE_DIM);
    const dim3 grid(
        (output_width + TILE_DIM - 1) / TILE_DIM,
        (output_height + TILE_DIM - 1) / TILE_DIM,
        batch_channels
    );

    const size_t shmem_size = (TILE_DIM * stride + (kernel_size - 1) * dilation) * 
                             (TILE_DIM * stride + (kernel_size - 1) * dilation) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward", ([&] {
        max_pool2d_shared_kernel<scalar_t><<<grid, threads, shmem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_channels,
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with shared memory (CUDA)");
}