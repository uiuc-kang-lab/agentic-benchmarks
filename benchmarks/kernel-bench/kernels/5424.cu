#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel using shared memory to store input tiles
template <typename scalar_t>
__global__ void max_pool2d_kernel_shared(
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
    // 2D indexing in block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * blockDim.x + tx;
    const int row = blockIdx.y * blockDim.y + ty;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    extern __shared__ scalar_t tile[];

    // Calculate input indices
    const int input_x = col * stride - padding + tx * dilation;
    const int input_y = row * stride - padding + ty * dilation;

    if (row < output_height && col < output_width) {
        // Copy necessary elements into shared memory
        if (input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height) {
            int input_idx = ((b * channels + c) * input_height + input_y) * input_width + input_x;
            tile[ty * blockDim.x + tx] = input[input_idx];
        }
        __syncthreads();

        // Calculate the maximum value from the shared tile
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int smem_idx = (ty + kh) * blockDim.x + (tx + kw);
                if (ty + kh < blockDim.y && tx + kw < blockDim.x) {
                    max_val = max(max_val, tile[smem_idx]);
                }
            }
        }

        // Store the maximum in the output
        int output_idx = ((b * channels + c) * output_height + row) * output_width + col;
        output[output_idx] = max_val;
    }
}

// Host function to launch the CUDA kernel
torch::Tensor max_pool2d_cuda_forward_shared(
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
    
    // Compute output dimensions
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // 2D block and grid configuration
    dim3 threads(16, 16);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    // Shared memory size
    const int shared_mem_size = threads.x * threads.y * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward_shared", ([&] {
        max_pool2d_kernel_shared<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &max_pool2d_cuda_forward_shared, "Max Pool 2D forward using shared memory (CUDA)");
}