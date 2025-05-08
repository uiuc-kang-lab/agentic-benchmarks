#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Constant memory for kernel parameters
__constant__ int c_kernel_size;
__constant__ int c_stride;
__constant__ int c_padding;
__constant__ int c_dilation;
__constant__ int c_dims[8];  // batch_size, channels, input_h, input_w, output_h, output_w, input_stride, output_stride

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output
) {
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= c_dims[0] * c_dims[1] * c_dims[4] * c_dims[5]) return;

    const int ow = output_idx % c_dims[5];
    const int oh = (output_idx / c_dims[5]) % c_dims[4];
    const int c = (output_idx / (c_dims[5] * c_dims[4])) % c_dims[1];
    const int b = output_idx / (c_dims[5] * c_dims[4] * c_dims[1]);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    const int input_batch_offset = b * c_dims[1] * c_dims[2] * c_dims[3];
    const int input_channel_offset = c * c_dims[2] * c_dims[3];

    #pragma unroll
    for (int kh = 0; kh < c_kernel_size; kh++) {
        const int ih = oh * c_stride - c_padding + kh * c_dilation;
        if (ih >= 0 && ih < c_dims[2]) {
            const int input_h_offset = ih * c_dims[3];
            
            #pragma unroll
            for (int kw = 0; kw < c_kernel_size; kw++) {
                const int iw = ow * c_stride - c_padding + kw * c_dilation;
                if (iw >= 0 && iw < c_dims[3]) {
                    const int input_idx = input_batch_offset + 
                                        input_channel_offset + 
                                        input_h_offset + 
                                        iw;
                    max_val = max(max_val, __ldg(&input[input_idx]));
                }
            }
        }
    }

    const int output_batch_stride = c_dims[1] * c_dims[4] * c_dims[5];
    const int output_channel_stride = c_dims[4] * c_dims[5];
    const int output_write_idx = b * output_batch_stride + 
                                c * output_channel_stride + 
                                oh * c_dims[5] + 
                                ow;
    output[output_write_idx] = max_val;
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

    // Copy constant parameters to device constant memory
    int dims[8] = {
        static_cast<int>(batch_size),
        static_cast<int>(channels),
        static_cast<int>(input_height),
        static_cast<int>(input_width),
        static_cast<int>(output_height),
        static_cast<int>(output_width),
        static_cast<int>(input_height * input_width),
        static_cast<int>(output_height * output_width)
    };
    
    cudaMemcpyToSymbol(c_kernel_size, &kernel_size, sizeof(int));
    cudaMemcpyToSymbol(c_stride, &stride, sizeof(int));
    cudaMemcpyToSymbol(c_padding, &padding, sizeof(int));
    cudaMemcpyToSymbol(c_dilation, &dilation, sizeof(int));
    cudaMemcpyToSymbol(c_dims, dims, sizeof(int) * 8);

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}