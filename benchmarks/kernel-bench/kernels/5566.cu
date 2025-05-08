#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Constant memory for kernel parameters and lookup tables
__constant__ int const_params[8];  // kernel_size, stride, padding, dilation
__constant__ int const_dims[6];    // batch_size, channels, input_h, input_w, output_h, output_w
__constant__ int const_strides[4]; // Precomputed strides for index calculations

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output
) {
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= const_dims[0] * const_dims[1] * const_dims[4] * const_dims[5]) return;

    // Use constant memory for frequently accessed values
    const int kernel_size = const_params[0];
    const int stride = const_params[1];
    const int padding = const_params[2];
    const int dilation = const_params[3];
    
    const int output_w = const_dims[5];
    const int output_h = const_dims[4];
    const int input_w = const_dims[3];
    const int input_h = const_dims[2];

    // Use precomputed strides from constant memory
    const int ow = output_idx % output_w;
    const int oh = (output_idx / output_w) % output_h;
    const int c = (output_idx / (output_w * output_h)) % const_dims[1];
    const int b = output_idx / (output_w * output_h * const_dims[1]);

    // Base input offset for current batch and channel
    const int base_input_offset = b * const_strides[0] + c * const_strides[1];
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    const int ih_base = oh * stride - padding;
    const int iw_base = ow * stride - padding;

    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = ih_base + kh * dilation;
        if (ih >= 0 && ih < input_h) {
            const int ih_offset = ih * input_w;
            
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = iw_base + kw * dilation;
                if (iw >= 0 && iw < input_w) {
                    const int input_idx = base_input_offset + ih_offset + iw;
                    max_val = max(max_val, __ldg(&input[input_idx]));
                }
            }
        }
    }

    output[output_idx] = max_val;
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

    // Setup constant memory for parameters
    const int params[8] = {kernel_size, stride, padding, dilation};
    const int dims[6] = {batch_size, channels, input_height, input_width, output_height, output_width};
    const int strides[4] = {
        channels * input_height * input_width,    // batch stride
        input_height * input_width,               // channel stride
        input_width,                              // height stride
        1                                         // width stride
    };

    cudaMemcpyToSymbol(const_params, params, sizeof(int) * 8);
    cudaMemcpyToSymbol(const_dims, dims, sizeof(int) * 6);
    cudaMemcpyToSymbol(const_strides, strides, sizeof(int) * 4);

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