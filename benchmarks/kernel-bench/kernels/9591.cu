#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Device functions for modular implementation
template <typename scalar_t>
__device__ __forceinline__ void load_input_tile(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ shared_mem,
    int n, int c, int h, int w,
    int in_height, int in_width,
    int in_channels, int shared_width) {
    
    if (h >= 0 && h < in_height && w >= 0 && w < in_width) {
        int x_idx = ((n * in_channels + c) * in_height + h) * in_width + w;
        shared_mem[threadIdx.y * shared_width + threadIdx.x] = x[x_idx];
    } else {
        shared_mem[threadIdx.y * shared_width + threadIdx.x] = 0;
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_conv_window(
    const scalar_t* __restrict__ shared_mem,
    const scalar_t* __restrict__ weight,
    int c, int kernel_size, int shared_width) {
    
    scalar_t sum = 0;
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            int sm_idx = (threadIdx.y + kh) * shared_width + (threadIdx.x + kw);
            int w_idx = (c * kernel_size + kh) * kernel_size + kw;
            sum += shared_mem[sm_idx] * weight[w_idx];
        }
    }
    return sum;
}

template <typename scalar_t>
__device__ __forceinline__ void write_output(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ bias,
    scalar_t conv_result,
    int n, int c, int h, int w,
    int out_height, int out_width,
    int in_channels) {
    
    if (h < out_height && w < out_width) {
        int out_idx = ((n * in_channels + c) * out_height + h) * out_width + w;
        out[out_idx] = conv_result + bias[c];
    }
}

template <typename scalar_t>
__global__ void depthwiseConv2DKernelModular(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {
    
    extern __shared__ char shared_memory[];
    scalar_t* shared_mem = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int block_x = blockIdx.x * blockDim.x;
    const int block_y = blockIdx.y * blockDim.y;
    const int global_x = block_x + threadIdx.x;
    const int global_y = block_y + threadIdx.y;
    const int bc_idx = blockIdx.z;
    const int c = bc_idx % in_channels;
    const int n = bc_idx / in_channels;
    const int in_x = global_x * stride - padding;
    const int in_y = global_y * stride - padding;
    const int shared_width = blockDim.x + kernel_size - 1;
    const int shared_height = blockDim.y + kernel_size - 1;
    
    if (threadIdx.x < shared_width && threadIdx.y < shared_height) {
        load_input_tile(x, shared_mem,
                       n, c, in_y, in_x,
                       in_height, in_width,
                       in_channels, shared_width);
    }
    __syncthreads();
    
    if (global_x < out_width && global_y < out_height) {
        scalar_t conv_result = compute_conv_window(shared_mem, w,
                                                 c, kernel_size, shared_width);
        write_output(out, b, conv_result,
                    n, c, global_y, global_x,
                    out_height, out_width, in_channels);
    }
}

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups) {
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());
    
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size * in_channels
    );
    
    const int shared_width = BLOCK_SIZE + kernel_size - 1;
    const int shared_height = BLOCK_SIZE + kernel_size - 1;
    const int shared_mem_size = shared_width * shared_height * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_modular", ([&] {
        depthwiseConv2DKernelModular<scalar_t><<<grid, block, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size, in_channels,
            in_height, in_width,
            kernel_size,
            out_height, out_width,
            stride, padding
        );
    }));
    
    return out;
}

torch::Tensor forward_wrap(
    torch::Tensor x,
    torch::Tensor weight,
    pybind11::object bias_obj,
    int stride,
    int padding,
    int groups) {
    
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
        "Modular depthwise conv2d forward",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}