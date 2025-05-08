#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t, int BLOCK_SIZE>
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
    __shared__ scalar_t shared_data[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int output_idx = blockIdx.x;
    
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    // Initialize local max value
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();

    // Calculate the starting positions for the pooling window
    const int start_h = oh * stride - padding;
    const int start_w = ow * stride - padding;

    // Each thread processes a portion of the kernel window
    for (int kh = tid; kh < kernel_size; kh += BLOCK_SIZE) {
        const int ih = start_h + kh * dilation;
        
        if (ih >= 0 && ih < input_height) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = start_w + kw * dilation;
                
                if (iw >= 0 && iw < input_width) {
                    const int input_idx = b * (channels * input_height * input_width) +
                                        c * (input_height * input_width) +
                                        ih * input_width + iw;
                    thread_max = max(thread_max, input[input_idx]);
                }
            }
        }
    }

    // Store thread's max value in shared memory
    shared_data[tid] = thread_max;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = BLOCK_SIZE/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    // Warp-level reduction using shuffle operations
    if (tid < 32) {
        scalar_t val = shared_data[tid];
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
        }

        if (tid == 0) {
            output[output_idx] = val;
        }
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

    const int total_elements = batch_size * channels * output_height * output_width;
    constexpr int BLOCK_SIZE = 256;
    const int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
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