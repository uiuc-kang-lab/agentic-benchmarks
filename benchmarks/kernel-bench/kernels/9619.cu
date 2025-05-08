#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Optimized block size for H100
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define ELEMENTS_PER_THREAD 4

template <typename scalar_t>
__global__ void depthwiseConv2DTunedKernel(
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
    
    extern __shared__ char shared_mem[];
    scalar_t* shared_weights = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_SIZE_X + tx;
    const int nc = blockIdx.z;
    const int c = nc % in_channels;
    const int n = nc / in_channels;
    
    // Collaborative loading of kernel weights into shared memory
    if (tid < kernel_size * kernel_size) {
        shared_weights[tid] = w[c * kernel_size * kernel_size + tid];
    }
    __syncthreads();
    
    // Calculate output positions
    const int out_x_base = blockIdx.x * BLOCK_SIZE_X * ELEMENTS_PER_THREAD + tx;
    const int out_y_base = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    // Process multiple elements per thread in x-direction
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int out_x = out_x_base + i * BLOCK_SIZE_X;
        
        if (out_x < out_width && out_y_base < out_height) {
            scalar_t sum = 0;
            
            // Pre-compute input positions
            const int in_y_start = out_y_base * stride - padding;
            const int in_x_start = out_x * stride - padding;
            
            // Unrolled convolution computation
            #pragma unroll
            for (int ky = 0; ky < kernel_size; ky++) {
                const int in_y = in_y_start + ky;
                const bool valid_y = (in_y >= 0 && in_y < in_height);
                
                #pragma unroll
                for (int kx = 0; kx < kernel_size; kx++) {
                    const int in_x = in_x_start + kx;
                    
                    if (valid_y && in_x >= 0 && in_x < in_width) {
                        const scalar_t input_val = x[((n * in_channels + c) * in_height + in_y) * in_width + in_x];
                        const scalar_t weight_val = shared_weights[ky * kernel_size + kx];
                        sum += input_val * weight_val;
                    }
                }
            }
            
            // Add bias and write output
            sum += b[c];
            out[((n * in_channels + c) * out_height + out_y_base) * out_width + out_x] = sum;
        }
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
    
    // Calculate grid dimensions
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);  // 128 threads per block
    dim3 grid(
        (out_width + BLOCK_SIZE_X * ELEMENTS_PER_THREAD - 1) / (BLOCK_SIZE_X * ELEMENTS_PER_THREAD),
        (out_height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        batch_size * in_channels
    );
    
    const int shared_mem_size = kernel_size * kernel_size * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_tuned", ([&] {
        depthwiseConv2DTunedKernel<scalar_t><<<grid, block, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            kernel_size,
            out_height,
            out_width,
            stride,
            padding
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
    m.def("forward",
          &forward_wrap,
          "Block-size optimized depthwise conv2d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("groups") = 1);
}