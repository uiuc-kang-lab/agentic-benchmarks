#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_shared_kernel(
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
    extern __shared__ scalar_t shared_block[];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int b = bz / channels;
    const int c = bz % channels;
    
    // Tile dimensions with padding for kernel
    const int TILE_W = 16;
    const int TILE_H = 16;
    const int SHMEM_W = TILE_W * stride + (KERNEL_SIZE - 1) * dilation;
    const int SHMEM_H = TILE_H * stride + (KERNEL_SIZE - 1) * dilation;

    // Load input tile into shared memory
    for (int i = ty; i < SHMEM_H; i += blockDim.y) {
        for (int j = tx; j < SHMEM_W; j += blockDim.x) {
            const int ih = by * TILE_H * stride - padding + i * dilation;
            const int iw = bx * TILE_W * stride - padding + j * dilation;
            
            scalar_t val = -std::numeric_limits<scalar_t>::infinity();
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                val = input[(b * channels + c) * input_height * input_width + ih * input_width + iw];
            }
            shared_block[i * SHMEM_W + j] = val;
        }
    }
    __syncthreads();

    // Compute output position
    const int oh = by * TILE_H + ty;
    const int ow = bx * TILE_W + tx;
    if (oh >= output_height || ow >= output_width) return;

    // Find max in shared memory tile
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int sh_start_h = ty * stride;
    const int sh_start_w = tx * stride;

    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
            const int sh_h = sh_start_h + kh * dilation;
            const int sh_w = sh_start_w + kw * dilation;
            if (sh_h < SHMEM_H && sh_w < SHMEM_W) {
                max_val = max(max_val, shared_block[sh_h * SHMEM_W + sh_w]);
            }
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    if (tx % 32 == 0) {
        output[(b * channels + c) * output_height * output_width + oh * output_width + ow] = max_val;
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

    const dim3 threads(32, 8);
    const dim3 blocks(
        (output_width + 15) / 16,
        (output_height + 15) / 16,
        batch_size * channels
    );

    const int shmem_size = (16 * stride + (kernel_size - 1) * dilation) * 
                          (16 * stride + (kernel_size - 1) * dilation) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_shared_kernel<scalar_t, 2><<<blocks, threads, shmem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else if (kernel_size == 3) {
            max_pool2d_shared_kernel<scalar_t, 3><<<blocks, threads, shmem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else {
            max_pool2d_shared_kernel<scalar_t, -1><<<blocks, threads, shmem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with shared memory (CUDA)");
}