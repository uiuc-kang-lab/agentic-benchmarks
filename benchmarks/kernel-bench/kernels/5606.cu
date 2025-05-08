#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size, int channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int stride, int padding, int dilation)
{
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_mem);

    const int b = blockIdx.z / channels;
    const int c = blockIdx.z % channels;
    
    const int tile_y = blockDim.y * blockIdx.y;
    const int tile_x = blockDim.x * blockIdx.x;
    
    const int shared_height = blockDim.y * stride + KERNEL_SIZE * dilation - stride + 1;
    const int shared_width = blockDim.x * stride + KERNEL_SIZE * dilation - stride + 1;

    const int load_threads = blockDim.x * blockDim.y;
    const int load_idx = threadIdx.y * blockDim.x + threadIdx.x;
    
    for(int i = load_idx; i < shared_height * shared_width; i += load_threads) {
        const int y = i / shared_width;
        const int x = i % shared_width;
        const int in_y = tile_y * stride - padding + y * dilation;
        const int in_x = tile_x * stride - padding + x * dilation;
        
        if(in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width)
            shared_input[y * shared_width + x] = input[
                b * channels * input_height * input_width +
                c * input_height * input_width +
                in_y * input_width + in_x];
        else
            shared_input[y * shared_width + x] = -std::numeric_limits<scalar_t>::infinity();
    }

    __syncthreads();

    const int y_out = tile_y + threadIdx.y;
    const int x_out = tile_x + threadIdx.x;
    if(y_out >= output_height || x_out >= output_width) return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    if constexpr (KERNEL_SIZE == 2) {
        #pragma unroll
        for(int ky = 0; ky < 2; ky++) {
            #pragma unroll
            for(int kx = 0; kx < 2; kx++) {
                const int y = threadIdx.y * stride + ky * dilation;
                const int x = threadIdx.x * stride + kx * dilation;
                max_val = fmaxf(max_val, shared_input[y * shared_width + x]);
            }
        }
    } else if constexpr (KERNEL_SIZE == 3) {
        #pragma unroll
        for(int ky = 0; ky < 3; ky++) {
            #pragma unroll
            for(int kx = 0; kx < 3; kx++) {
                const int y = threadIdx.y * stride + ky * dilation;
                const int x = threadIdx.x * stride + kx * dilation;
                max_val = fmaxf(max_val, shared_input[y * shared_width + x]);
            }
        }
    } else {
        for(int ky = 0; ky < KERNEL_SIZE; ky++) {
            for(int kx = 0; kx < KERNEL_SIZE; kx++) {
                const int y = threadIdx.y * stride + ky * dilation;
                const int x = threadIdx.x * stride + kx * dilation;
                max_val = fmaxf(max_val, shared_input[y * shared_width + x]);
            }
        }
    }

    output[b * channels * output_height * output_width +
           c * output_height * output_width +
           y_out * output_width + x_out] = max_val;
}

torch::Tensor max_pool2d_cuda_shared_forward(
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

    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    constexpr int BLOCK = 16;
    const int shared_mem_size = (BLOCK * stride + kernel_size * dilation) * 
                               (BLOCK * stride + kernel_size * dilation) * sizeof(float);

    dim3 blocks(
        (output_width + BLOCK - 1) / BLOCK,
        (output_height + BLOCK - 1) / BLOCK,
        batch_size * channels
    );
    dim3 threads(BLOCK, BLOCK);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward_shared", ([&] {
        if(kernel_size == 2) {
            max_pool2d_shared_kernel<scalar_t, 2><<<blocks, threads, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else if(kernel_size == 3) {
            max_pool2d_shared_kernel<scalar_t, 3><<<blocks, threads, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else {
            max_pool2d_shared_kernel<scalar_t, -1><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &max_pool2d_cuda_shared_forward, "Max Pool 2D forward with shared memory (CUDA)");
}