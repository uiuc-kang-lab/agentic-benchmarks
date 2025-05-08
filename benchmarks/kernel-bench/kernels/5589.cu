#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int c_kernel_size;
__constant__ int c_stride;
__constant__ int c_padding;
__constant__ int c_dilation;

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_balanced_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    // 2D block for output height and width
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process multiple channels per thread block
    const int c_start = blockIdx.z * blockDim.z;
    const int c_step = blockDim.z * gridDim.z;
    
    if (oh >= output_height || ow >= output_width) return;
    
    // Pre-compute input position bases
    const int ih_base = oh * c_stride - c_padding;
    const int iw_base = ow * c_stride - c_padding;
    
    // Process multiple channels in this thread
    for (int b = 0; b < batch_size; b++) {
        const int batch_offset = b * channels * input_height * input_width;
        
        for (int c = c_start; c < channels; c += c_step) {
            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
            const int channel_offset = c * input_height * input_width;
            
            if constexpr (KERNEL_SIZE > 0) {
                #pragma unroll
                for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                    const int ih = ih_base + kh * c_dilation;
                    if (ih >= 0 && ih < input_height) {
                        const int row_offset = ih * input_width;
                        
                        #pragma unroll
                        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                            const int iw = iw_base + kw * c_dilation;
                            if (iw >= 0 && iw < input_width) {
                                const int idx = batch_offset + channel_offset + row_offset + iw;
                                max_val = max(max_val, __ldg(&input[idx]));
                            }
                        }
                    }
                }
            } else {
                for (int kh = 0; kh < c_kernel_size; kh++) {
                    const int ih = ih_base + kh * c_dilation;
                    if (ih >= 0 && ih < input_height) {
                        const int row_offset = ih * input_width;
                        
                        for (int kw = 0; kw < c_kernel_size; kw++) {
                            const int iw = iw_base + kw * c_dilation;
                            if (iw >= 0 && iw < input_width) {
                                const int idx = batch_offset + channel_offset + row_offset + iw;
                                max_val = max(max_val, __ldg(&input[idx]));
                            }
                        }
                    }
                }
            }
            
            const int out_idx = b * channels * output_height * output_width +
                               c * output_height * output_width +
                               oh * output_width + ow;
            output[out_idx] = max_val;
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

    // Copy constants to constant memory
    cudaMemcpyToSymbol(c_kernel_size, &kernel_size, sizeof(int));
    cudaMemcpyToSymbol(c_stride, &stride, sizeof(int));
    cudaMemcpyToSymbol(c_padding, &padding, sizeof(int));
    cudaMemcpyToSymbol(c_dilation, &dilation, sizeof(int));

    // Configure grid and block dimensions for better occupancy
    dim3 threads(16, 16, 1);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        min(32, (channels + 3) / 4)
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_balanced_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                batch_size, channels, input_height, input_width,
                output_height, output_width);
        } else if (kernel_size == 3) {
            max_pool2d_balanced_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                batch_size, channels, input_height, input_width,
                output_height, output_width);
        } else {
            max_pool2d_balanced_kernel<scalar_t, 0><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                batch_size, channels, input_height, input_width,
                output_height, output_width);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}