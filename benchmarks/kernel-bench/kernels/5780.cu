#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Store pooling parameters: [0]: kernel_size, [1]: stride, [2]: padding, [3]: dilation
// This constant memory is read-only and shared by all threads
__constant__ int pool_const[4];

// Kernel for Max Pooling 2D using constant memory for pooling parameters
// Input tensor is expected in NCHW format (batch, channels, input_height, input_width)
// Output tensor is in NCHW format as well

template <typename scalar_t>
__global__ void max_pool2d_const_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_channels,  // equals batch_size * channels
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    // Read pooling parameters from constant memory
    const int k       = pool_const[0];
    const int stride  = pool_const[1];
    const int padding = pool_const[2];
    const int dilation= pool_const[3];

    const int output_elements = output_height * output_width; // per channel element
    const int total_elements = batch_channels * output_elements;

    // Grid-stride loop to cover each output element
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const int bc = idx / output_elements;  // combined batch and channel index
        const int out_index = idx % output_elements;
        const int oh = out_index / output_width;
        const int ow = out_index % output_width;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        if (k == 2) {
            // Unrolled loops for k = 2
            #pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 2; ++kw) {
                    const int ih = oh * stride - padding + kh * dilation;
                    const int iw = ow * stride - padding + kw * dilation;
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        const int input_idx = (bc * input_height + ih) * input_width + iw;
                        max_val = max(max_val, input[input_idx]);
                    }
                }
            }
        } else {
            // General case for arbitrary kernel size
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    const int ih = oh * stride - padding + kh * dilation;
                    const int iw = ow * stride - padding + kw * dilation;
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        const int input_idx = (bc * input_height + ih) * input_width + iw;
                        max_val = max(max_val, input[input_idx]);
                    }
                }
            }
        }
        output[idx] = max_val;
    }
}

// Host wrapper that copies pooling parameters to constant memory and launches the kernel

torch::Tensor max_pool2d_cuda_forward_const(
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
    const int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Copy the pooling parameters to constant memory
    int host_pool_params[4] = { kernel_size, stride, padding, dilation };
    cudaMemcpyToSymbol(pool_const, host_pool_params, sizeof(host_pool_params));

    const int batch_channels = batch_size * channels;
    const int threads = 256;
    const int total_output = batch_channels * output_height * output_width;
    const int blocks = (total_output + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_const", ([&] {
        max_pool2d_const_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_channels,
            input_height,
            input_width,
            output_height,
            output_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward_const, "Max Pool 2D forward with constant memory optimization (CUDA)");
}
