#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define TILE_SIZE 16
#define BLOCK_SIZE 16
#define CHANNELS_PER_BLOCK 32

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k, int stride, int padding, int dilation) {

    __shared__ scalar_t shared_input[TILE_SIZE + 2][TILE_SIZE + 2];
    __shared__ scalar_t shared_weight[CHANNELS_PER_BLOCK][3][3];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int z = blockIdx.z;
    const int batch_idx = z / ((channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK);
    const int channel_block = (z % ((channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK)) * CHANNELS_PER_BLOCK;

    // Load weights into shared memory
    if (ty < 3 && tx < 3) {
        for (int c = 0; c < CHANNELS_PER_BLOCK; c++) {
            int channel = channel_block + c;
            if (channel < channels) {
                shared_weight[c][ty][tx] = weight[channel * k * k + ty * k + tx];
            }
        }
    }
    __syncthreads();

    // Process TILE_SIZE x TILE_SIZE output elements
    for (int c = 0; c < CHANNELS_PER_BLOCK; c++) {
        int channel = channel_block + c;
        if (channel >= channels) break;

        // Load input tile into shared memory
        int in_y = by + ty - padding;
        int in_x = bx + tx - padding;
        
        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
            shared_input[ty][tx] = input[
                batch_idx * channels * in_h * in_w +
                channel * in_h * in_w +
                in_y * in_w + in_x];
        } else {
            shared_input[ty][tx] = 0;
        }
        __syncthreads();

        // Compute output
        if (ty < TILE_SIZE && tx < TILE_SIZE) {
            int out_y = by + ty;
            int out_x = bx + tx;
            
            if (out_y < out_h && out_x < out_w) {
                scalar_t sum = 0;
                
                #pragma unroll
                for (int i = 0; i < k; i++) {
                    #pragma unroll
                    for (int j = 0; j < k; j++) {
                        int sy = ty + i;
                        int sx = tx + j;
                        sum += shared_input[sy][sx] * shared_weight[c][i][j];
                    }
                }
                
                if (bias != nullptr) {
                    sum += bias[channel];
                }
                
                output[
                    batch_idx * channels * out_h * out_w +
                    channel * out_h * out_w +
                    out_y * out_w + out_x] = sum;
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch, int in_channels, int out_channels,
    int height, int width) {

    __shared__ scalar_t shared_input[BLOCK_SIZE][BLOCK_SIZE][CHANNELS_PER_BLOCK];
    __shared__ scalar_t shared_weight[CHANNELS_PER_BLOCK][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_SIZE;
    const int by = blockIdx.y * BLOCK_SIZE;
    const int batch_idx = blockIdx.z / ((out_channels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const int out_channel_block = (blockIdx.z % ((out_channels + BLOCK_SIZE - 1) / BLOCK_SIZE)) * BLOCK_SIZE;

    scalar_t sum = 0;

    // Process input channels in tiles
    for (int ic = 0; ic < in_channels; ic += CHANNELS_PER_BLOCK) {
        // Load input tile into shared memory
        if (bx + tx < width && by + ty < height) {
            for (int c = 0; c < CHANNELS_PER_BLOCK; c++) {
                if (ic + c < in_channels) {
                    shared_input[ty][tx][c] = input[
                        batch_idx * in_channels * height * width +
                        (ic + c) * height * width +
                        (by + ty) * width + (bx + tx)];
                }
            }
        }

        // Load weights into shared memory
        if (ty < CHANNELS_PER_BLOCK && tx < BLOCK_SIZE) {
            int out_c = out_channel_block + tx;
            if (out_c < out_channels && ic + ty < in_channels) {
                shared_weight[ty][tx] = weight[out_c * in_channels + ic + ty];
            }
        }
        __syncthreads();

        // Compute partial sum
        if (bx + tx < width && by + ty < height) {
            for (int c = 0; c < CHANNELS_PER_BLOCK && ic + c < in_channels; c++) {
                sum += shared_input[ty][tx][c] * shared_weight[c][threadIdx.x];
            }
        }
        __syncthreads();
    }

    // Write output
    if (bx + tx < width && by + ty < height) {
        int out_c = out_channel_block + threadIdx.x;
        if (out_c < out_channels) {
            if (bias != nullptr) {
                sum += bias[out_c];
            }
            output[
                batch_idx * out_channels * height * width +
                out_c * height * width +
                (by + ty) * width + (bx + tx)] = sum;
        }
    }
}

torch::Tensor forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride, int padding, int dilation) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(depthwise_weight.is_cuda(), "Depthwise weight must be a CUDA tensor");
    TORCH_CHECK(pointwise_weight.is_cuda(), "Pointwise weight must be a CUDA tensor");

    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int k = depthwise_weight.size(2);
    int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

    auto depthwise_output = torch::empty({batch, channels, out_h, out_w}, x.options());

    dim3 depthwise_block(TILE_SIZE, TILE_SIZE);
    dim3 depthwise_grid(
        (out_w + TILE_SIZE - 1) / TILE_SIZE,
        (out_h + TILE_SIZE - 1) / TILE_SIZE,
        batch * ((channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK)
    );

    const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0) ?
        depthwise_bias.data_ptr() : nullptr;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<depthwise_grid, depthwise_block>>>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            reinterpret_cast<const scalar_t*>(depthwise_bias_ptr),
            depthwise_output.data_ptr<scalar_t>(),
            batch, channels,
            in_h, in_w,
            out_h, out_w,
            k, stride, padding, dilation);
    }));

    int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

    dim3 pointwise_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 pointwise_grid(
        (out_w + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch * ((out_channels + BLOCK_SIZE - 1) / BLOCK_SIZE)
    );

    const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0) ?
        pointwise_bias.data_ptr() : nullptr;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
        pointwise_conv2d_kernel<scalar_t><<<pointwise_grid, pointwise_block>>>(
            depthwise_output.data_ptr<scalar_t>(),
            pointwise_weight.data_ptr<scalar_t>(),
            reinterpret_cast<const scalar_t*>(pointwise_bias_ptr),
            output.data_ptr<scalar_t>(),
            batch, channels, out_channels,
            out_h, out_w);
    }));

    return output;
}

at::Tensor toTensor(const py::object& obj) {
    if (obj.is_none()) {
        return at::Tensor();
    }
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
    int stride, int padding, int dilation) {

    auto x = toTensor(x_obj);
    auto depthwise_weight = toTensor(depthwise_weight_obj);
    auto pointwise_weight = toTensor(pointwise_weight_obj);
    auto depthwise_bias = toTensor(depthwise_bias_obj);
    auto pointwise_bias = toTensor(pointwise_bias_obj);

    return forward_cuda(
        x, depthwise_weight, pointwise_weight,
        depthwise_bias, pointwise_bias,
        stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward with shared memory optimization");
}