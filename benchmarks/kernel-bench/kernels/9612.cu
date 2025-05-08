#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Define output tile dimensions
#define TOUT_W 16
#define TOUT_H 16
#define BLOCK_DIM 16

// Optimized depthwise convolution kernel using warp-level primitives for reduction
template <typename scalar_t>
__global__ void depthwiseConv2DWarpOptimizedKernel(
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
    scalar_t* smem_weight = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* smem_input = smem_weight + kernel_size * kernel_size;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nc = blockIdx.z;
    int c = nc % in_channels;
    int n = nc / in_channels;

    int tile_out_x = blockIdx.x * blockDim.x;
    int tile_out_y = blockIdx.y * blockDim.y;

    int tile_in_x = tile_out_x * stride - padding;
    int tile_in_y = tile_out_y * stride - padding;

    int tile_in_w = (TOUT_W - 1) * stride + kernel_size;
    int tile_in_h = (TOUT_H - 1) * stride + kernel_size;

    int weight_elems = kernel_size * kernel_size;
    for (int i = tid; i < weight_elems; i += blockDim.x * blockDim.y) {
        smem_weight[i] = w[c * weight_elems + i];
    }

    int input_elems = tile_in_w * tile_in_h;
    for (int i = tid; i < input_elems; i += blockDim.x * blockDim.y) {
        int tx = i % tile_in_w;
        int ty = i / tile_in_w;
        int in_x = tile_in_x + tx;
        int in_y = tile_in_y + ty;
        scalar_t val = 0;
        if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
            int idx = ((n * in_channels + c) * in_height + in_y) * in_width + in_x;
            val = x[idx];
        }
        smem_input[ty * tile_in_w + tx] = val;
    }

    __syncthreads();

    int tx_out = threadIdx.x;
    int ty_out = threadIdx.y;
    int out_x = tile_out_x + tx_out;
    int out_y = tile_out_y + ty_out;

    if (tx_out < TOUT_W && ty_out < TOUT_H && out_x < out_width && out_y < out_height) {
        scalar_t sum = b[c];
        int in_offset_x = tx_out * stride;
        int in_offset_y = ty_out * stride;

        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                sum += smem_input[(in_offset_y + i) * tile_in_w + (in_offset_x + j)] * smem_weight[i * kernel_size + j];
            }
        }

        // Warp-level reduction
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (tx_out % warpSize == 0) {
            int out_idx = ((n * in_channels + c) * out_height + out_y) * out_width + out_x;
            out[out_idx] = sum;
        }
    }
}

// Forward implementation wrapping the kernel launch
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

    dim3 block(TOUT_W, TOUT_H);
    dim3 grid((out_width + TOUT_W - 1) / TOUT_W, (out_height + TOUT_H - 1) / TOUT_H, batch_size * in_channels);

    int tile_in_w = (TOUT_W - 1) * stride + kernel_size;
    int tile_in_h = (TOUT_H - 1) * stride + kernel_size;
    size_t shmem_size = (kernel_size * kernel_size + tile_in_w * tile_in_h) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_warp_optimized", ([&] {
        depthwiseConv2DWarpOptimizedKernel<scalar_t><<<grid, block, shmem_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size, in_channels, in_height, in_width, kernel_size, out_height, out_width, stride, padding
        );
    }));

    return out;
}

namespace py = pybind11;

// Wrapper to handle optional bias from Python
torch::Tensor forward_wrap(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
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
          "Optimized depthwise conv2d with warp-level reduction",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("groups") = 1);
}
