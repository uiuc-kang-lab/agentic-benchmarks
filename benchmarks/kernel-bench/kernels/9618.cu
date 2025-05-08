#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

// Device function to load weights into shared memory
__device__ void load_weights_to_shared(const float* __restrict__ w, float* shared_weights, int c, int kernel_size) {
    int tid = threadIdx.x;
    int weight_elems = kernel_size * kernel_size;
    if (tid < weight_elems) {
        shared_weights[tid] = w[c * weight_elems + tid];
    }
}

// Device function to compute convolution for a single output element
__device__ float compute_convolution(const float* __restrict__ x, const float* shared_weights, int n, int c, int in_height, int in_width, int kernel_size, int stride, int padding, int out_x, int out_y) {
    int in_y_base = out_y * stride - padding;
    int in_x_base = out_x * stride - padding;
    float sum = 0;
    #pragma unroll
    for (int ky = 0; ky < kernel_size; ky++) {
        int in_y = in_y_base + ky;
        bool valid_y = (in_y >= 0 && in_y < in_height);
        #pragma unroll
        for (int kx = 0; kx < kernel_size; kx++) {
            int in_x = in_x_base + kx;
            if (valid_y && in_x >= 0 && in_x < in_width) {
                float input_val = x[((n * in_channels + c) * in_height + in_y) * in_width + in_x];
                float weight_val = shared_weights[ky * kernel_size + kx];
                sum += input_val * weight_val;
            }
        }
    }
    return sum;
}

// Kernel function
__global__ void depthwiseConv2DModularKernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {
    extern __shared__ float shared_weights[];
    int nc = blockIdx.z;
    int c = nc % in_channels;
    int n = nc / in_channels;
    load_weights_to_shared(w, shared_weights, c, kernel_size);
    __syncthreads();
    int elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    int total_elements = out_height * out_width;
    int base_idx = blockIdx.y * elements_per_block;
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int element_idx = base_idx + threadIdx.x + i * BLOCK_SIZE;
        if (element_idx >= total_elements) continue;
        int out_y = element_idx / out_width;
        int out_x = element_idx % out_width;
        float sum = compute_convolution(x, shared_weights, n, c, in_height, in_width, kernel_size, stride, padding, out_x, out_y);
        if (element_idx < total_elements) {
            sum += b[c];
            out[((n * in_channels + c) * out_height + out_y) * out_width + out_x] = sum;
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
    const int total_elements = out_height * out_width;
    const int elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int num_blocks_y = (total_elements + elements_per_block - 1) / elements_per_block;
    dim3 grid(1, num_blocks_y, batch_size * in_channels);
    dim3 block(BLOCK_SIZE);
    const int shared_mem_size = kernel_size * kernel_size * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_modular", ([&] {
        depthwiseConv2DModularKernel<<<grid, block, shared_mem_size>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
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
          "Modular depthwise conv2d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("groups") = 1);
}