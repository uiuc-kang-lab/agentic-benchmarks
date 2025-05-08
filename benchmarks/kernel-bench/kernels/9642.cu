#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Define constant memory for kernel weights and bias
__constant__ float const_weights[3*3*512];  // Assuming max 512 channels and 3x3 kernel
__constant__ float const_bias[512];         // Assuming max 512 channels

template <typename scalar_t>
__global__ void depthwiseConv2DKernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding)
{
    // Calculate global output indices
    const int w_out_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int h_out_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const int c = blockIdx.z % in_channels;
    const int n = blockIdx.z / in_channels;

    // Check bounds for output
    if (w_out_idx >= out_width || h_out_idx >= out_height || n >= batch_size) {
        return;
    }

    // Dimensions of output tile in this block
    const int tile_out_w = blockDim.x;
    const int tile_out_h = blockDim.y;
    // Compute the dimensions of the input tile required for this block.
    // For each output pixel, we need a region of size kernel_size; with stride, the full tile is:
    const int tile_in_w = tile_out_w * stride + (kernel_size - stride);
    const int tile_in_h = tile_out_h * stride + (kernel_size - stride);

    // Top-left coordinate in the input for this block
    const int in_x_origin = blockIdx.x * blockDim.x * stride - padding;
    const int in_y_origin = blockIdx.y * blockDim.y * stride - padding;

    // Declare shared memory with extra padding to alleviate bank conflicts
    extern __shared__ scalar_t shmem[];
    // Using a pitch for shared memory to avoid bank conflicts
    const int shmem_pitch = tile_in_w + 1;

    // Each thread loads one or more elements into shared memory for the input tile
    const int sh_size = tile_in_w * tile_in_h;
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < sh_size; idx += blockDim.x * blockDim.y) {
        int r = idx / tile_in_w;
        int c_sh = idx % tile_in_w;
        int g_y = in_y_origin + r;
        int g_x = in_x_origin + c_sh;
        int sh_idx = r * shmem_pitch + c_sh;
        if (g_y >= 0 && g_y < in_height && g_x >= 0 && g_x < in_width) {
            int x_index = ((n * in_channels + c) * in_height + g_y) * in_width + g_x;
            shmem[sh_idx] = x[x_index];
        } else {
            shmem[sh_idx] = 0;
        }
    }

    __syncthreads();

    // Each thread computes one output element using data from shared memory
    scalar_t value = 0;
    // The starting coordinate in shared memory for this output pixel
    int local_y = threadIdx.y * stride;
    int local_x = threadIdx.x * stride;

    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            int sh_idx = (local_y + kh) * shmem_pitch + (local_x + kw);
            int weight_idx = (c * kernel_size + kh) * kernel_size + kw;
            value += shmem[sh_idx] * const_weights[weight_idx];
        }
    }

    value += const_bias[c];
    
    int out_idx = ((n * in_channels + c) * out_height + h_out_idx) * out_width + w_out_idx;
    out[out_idx] = value;
}

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups)
{
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Copy weights and bias to constant memory
    cudaMemcpyToSymbol(const_weights, weight.data_ptr<float>(), 
                       weight.numel() * sizeof(float));
    cudaMemcpyToSymbol(const_bias, bias.data_ptr<float>(), 
                       bias.numel() * sizeof(float));

    // Use 32x8 thread block configuration for better memory coalescing
    dim3 threads(32, 8);
    dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size * in_channels
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward", ([&] {
        depthwiseConv2DKernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size, in_channels, in_height, in_width,
            kernel_size, out_height, out_width,
            stride, padding
        );
    }));

    return out;
}

namespace py = pybind11;

torch::Tensor forward_wrap(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    int stride,
    int padding,
    int groups)
{
    torch::Tensor bias;
    if (bias_obj.is_none()) {
        bias = torch::zeros({x.size(1)}, x.options());
    } else {
        bias = bias_obj.cast<torch::Tensor>();
    }
    return forward_impl(x, weight, bias, stride, padding, groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward_wrap,
        "Depthwise conv2d forward with constant memory optimization",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}