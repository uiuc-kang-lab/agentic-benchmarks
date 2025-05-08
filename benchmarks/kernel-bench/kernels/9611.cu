#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Define optimal tile dimensions
#define TILE_W 32
#define TILE_H 8

// Unified depthwise convolution kernel using optimized indexing and shared memory
// with effective tile and block management

template <typename scalar_t>
__global__ void depthwiseConv2DCombined(
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

    // Block index determines batch and channel
    const int nc = blockIdx.z;
    const int c = nc % in_channels;
    const int n = nc / in_channels;

    // Compute output tile starting indices
    const int out_tile_x = blockIdx.x * TILE_W;
    const int out_tile_y = blockIdx.y * TILE_H;

    // Index of this thread in the block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Determine shared memory tile dimensions
    const int shared_width = (TILE_W - 1) * stride + kernel_size;
    const int shared_height = (TILE_H - 1) * stride + kernel_size;

    // Allocate shared memory for input tile and filter weights
    extern __shared__ char shared_memory[];
    scalar_t* shmem_input = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* shmem_weight = shmem_input + (shared_width * shared_height);

    // Load filter weights into shared memory cooperatively
    if (tx == 0 && ty == 0) {  // Only one thread loads weights
        #pragma unroll
        for (int i = 0; i < kernel_size * kernel_size; ++i) {
            shmem_weight[i] = w[c * kernel_size * kernel_size + i];
        }
    }
    
    // Calculate input tile starting coordinates
    const int in_tile_x = out_tile_x * stride - padding;
    const int in_tile_y = out_tile_y * stride - padding;

    // Load the shared memory tile cooperatively
    for (int j = ty; j < shared_height; j += blockDim.y) {
        for (int i = tx; i < shared_width; i += blockDim.x) {
            int in_x = in_tile_x + i;
            int in_y = in_tile_y + j;
            scalar_t val = 0;
            if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                int input_idx = ((n * in_channels + c) * in_height + in_y) * in_width + in_x;
                val = x[input_idx];
            }
            shmem_input[j * shared_width + i] = val;
        }
    }
    __syncthreads();

    // Compute output tile elements assigned to this thread
    const int out_x = out_tile_x + tx;
    const int out_y = out_tile_y + ty;

    if (tx < TILE_W && ty < TILE_H && out_x < out_width && out_y < out_height) {
        scalar_t sum = 0;
        int sh_x = tx * stride;
        int sh_y = ty * stride;
        
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                int shared_idx = (sh_y + kh) * shared_width + (sh_x + kw);
                int weight_idx = kh * kernel_size + kw;
                sum += shmem_input[shared_idx] * shmem_weight[weight_idx];
            }
        }
        sum += b[c];
        int out_idx = ((n * in_channels + c) * out_height + out_y) * out_width + out_x;
        out[out_idx] = sum;
    }
}

// Forward implementation wrapping the combined kernel

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
    const int in_width  = x.size(3);
    
    const int kernel_size = weight.size(2);
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Grid setup for optimal performance
    dim3 block(TILE_W, TILE_H);
    dim3 grid((out_width + TILE_W - 1) / TILE_W, (out_height + TILE_H - 1) / TILE_H, batch_size * in_channels);

    // Determine shared memory size needed
    int shared_width = (TILE_W - 1) * stride + kernel_size;
    int shared_height = (TILE_H - 1) * stride + kernel_size;
    size_t shmem_size = (shared_width * shared_height + kernel_size * kernel_size) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_combined", ([&] {
        depthwiseConv2DCombined<scalar_t><<<grid, block, shmem_size>>>(
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

// Wrapper to handle optional bias from Python

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

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward_wrap,
        "Combined depthwise conv2d forward with optimized memory and indexing",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
