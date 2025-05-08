#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define BLOCK_SIZE 256
#define TILE_DIM 16
#define ELEMENTS_PER_THREAD 4

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k, int stride,
    int padding, int dilation) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * channels * out_h * out_w;
    
    if (idx >= total) return;

    const int ow = idx % out_w;
    const int oh = (idx / out_w) % out_h;
    const int c = (idx / (out_w * out_h)) % channels;
    const int n = idx / (out_w * out_h * channels);

    scalar_t sum = 0;
    #pragma unroll
    for (int i = 0; i < k; ++i) {
        #pragma unroll
        for (int j = 0; j < k; ++j) {
            const int ih = oh * stride - padding + i * dilation;
            const int iw = ow * stride - padding + j * dilation;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                sum += input[n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw] *
                       weight[c * k * k + i * k + j];
            }
        }
    }
    if (bias != nullptr) {
        sum += bias[c];
    }
    output[idx] = sum;
}

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width) {

    __shared__ scalar_t shared_input[TILE_DIM][TILE_DIM];
    __shared__ scalar_t shared_weight[TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int batch_idx = bz / ((out_channels + TILE_DIM - 1) / TILE_DIM);
    const int out_ch_block = (bz % ((out_channels + TILE_DIM - 1) / TILE_DIM)) * TILE_DIM;

    #pragma unroll
    for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
        const int out_x = bx * TILE_DIM * ELEMENTS_PER_THREAD + tx + e * TILE_DIM;
        if (out_x >= width) continue;

        scalar_t out_vals[TILE_DIM] = {0};

        for (int tile = 0; tile < (in_channels + TILE_DIM - 1) / TILE_DIM; ++tile) {
            if (ty < TILE_DIM && (tile * TILE_DIM + tx) < in_channels) {
                shared_input[ty][tx] = input[batch_idx * in_channels * height * width +
                                           (tile * TILE_DIM + tx) * height * width +
                                           by * TILE_DIM * width + out_x];
            }

            if (ty < TILE_DIM && (out_ch_block + ty) < out_channels &&
                (tile * TILE_DIM + tx) < in_channels) {
                shared_weight[ty][tx] = weight[(out_ch_block + ty) * in_channels +
                                             tile * TILE_DIM + tx];
            }
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < TILE_DIM; ++k) {
                if ((tile * TILE_DIM + k) < in_channels) {
                    #pragma unroll
                    for (int oc = 0; oc < TILE_DIM; ++oc) {
                        if ((out_ch_block + oc) < out_channels) {
                            out_vals[oc] += shared_input[ty][k] * shared_weight[oc][k];
                        }
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int oc = 0; oc < TILE_DIM; ++oc) {
            if ((out_ch_block + oc) < out_channels && out_x < width) {
                const int out_idx = batch_idx * out_channels * height * width +
                                  (out_ch_block + oc) * height * width +
                                  by * TILE_DIM * width + out_x;
                if (bias != nullptr) {
                    output[out_idx] = out_vals[oc] + bias[out_ch_block + oc];
                } else {
                    output[out_idx] = out_vals[oc];
                }
            }
        }
    }
}

torch::Tensor forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride,
    int padding,
    int dilation) {

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int k = depthwise_weight.size(2);
    const int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

    auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());

    const int total_threads_dw = batch * in_channels * out_h * out_w;
    const int blocks_dw = (total_threads_dw + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks_dw, BLOCK_SIZE>>>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            depthwise_bias.defined() ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
            depthwise_output.data_ptr<scalar_t>(),
            batch, in_channels, in_h, in_w, out_h, out_w,
            k, stride, padding, dilation);
    }));

    const int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks(
        (out_w + TILE_DIM * ELEMENTS_PER_THREAD - 1) / (TILE_DIM * ELEMENTS_PER_THREAD),
        (out_h + TILE_DIM - 1) / TILE_DIM,
        batch * ((out_channels + TILE_DIM - 1) / TILE_DIM)
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
        pointwise_conv2d_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            depthwise_output.data_ptr<scalar_t>(),
            pointwise_weight.data_ptr<scalar_t>(),
            pointwise_bias.defined() ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch, in_channels, out_channels, out_h, out_w);
    }));

    return output;
}

at::Tensor toTensor(const py::object& obj) {
    if (obj.is_none()) return at::Tensor();
    try {
        return obj.cast<at::Tensor>();
    } catch (const py::cast_error& e) {
        if (py::hasattr(obj, "data")) {
            return obj.attr("data").cast<at::Tensor>();
        }
        throw std::runtime_error("Expected a torch Tensor or Parameter.");
    }
}

at::Tensor forward_wrapper(
    py::object x_obj,
    py::object depthwise_weight_obj,
    py::object pointwise_weight_obj,
    py::object depthwise_bias_obj,
    py::object pointwise_bias_obj,
    int stride,
    int padding,
    int dilation) {

    return forward_cuda(
        toTensor(x_obj),
        toTensor(depthwise_weight_obj),
        toTensor(pointwise_weight_obj),
        toTensor(depthwise_bias_obj),
        toTensor(pointwise_bias_obj),
        stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward");
}