#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_sharedmem_kernel(
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
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_mem);

    const int BLOCK_W = 16;
    const int BLOCK_H = 16;

    const int ow = blockIdx.x * BLOCK_W + threadIdx.x;
    const int oh = blockIdx.y * BLOCK_H + threadIdx.y;
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;

    const int input_tile_start_oh = oh * stride - padding;
    const int input_tile_start_ow = ow * stride - padding;
    
    const int input_tile_h = BLOCK_H * stride + (kernel_size - 1) * dilation;
    const int input_tile_w = BLOCK_W * stride + (kernel_size - 1) * dilation;
    const int shared_size = input_tile_h * input_tile_w;

    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;
    
    // Load input tile into shared memory
    for (int i = threadIdx.y * BLOCK_W + threadIdx.x; i < shared_size; i += BLOCK_W * BLOCK_H) {
        const int kh = i / input_tile_w;
        const int kw = i % input_tile_w;
        const int ih = input_tile_start_oh + kh * dilation;
        const int iw = input_tile_start_ow + kw * dilation;
        
        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            shared_input[i] = input_channel[ih * input_width + iw];
        } else {
            shared_input[i] = -std::numeric_limits<scalar_t>::infinity();
        }
    }
    __syncthreads();

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    if (oh < output_height && ow < output_width) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int shmem_h = threadIdx.y * stride + kh * dilation;
                const int shmem_w = threadIdx.x * stride + kw * dilation;
                if (shmem_h < input_tile_h && shmem_w < input_tile_w) {
                    max_val = max(max_val, shared_input[shmem_h * input_tile_w + shmem_w]);
                }
            }
        }
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

    const dim3 threads(16, 16);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    const int shared_mem_size = (threads.x * stride + (kernel_size-1)*dilation) * 
                                (threads.y * stride + (kernel_size-1)*dilation) * 
                                sizeof(typename c10::cuda::c10_float16);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_sharedmem_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
