#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <algorithm>

// Kernel for depthwise 2D convolution for a tile (a subset of the batch).
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,    // number of batches in this tile
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int stride,
    int padding,
    int dilation) {

    // Compute output coordinate within the tile
    int ow = threadIdx.x + blockIdx.x * blockDim.x;
    int oh = blockIdx.y % out_h;
    int c  = blockIdx.y / out_h;
    int b  = blockIdx.z; // index within the current tile

    if (ow < out_w && c < channels && b < batch) {
        float sum = 0.0f;
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding; // kernel width is 1
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                int weight_idx = c * kernel_h + kh;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
        sum += bias[c];
        int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[output_idx] = sum;
    }
}

// Forward function using CUDA streams to overlap computation and (potential) memory operations
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    int num_streams = 2,
    int block_size = 256) {

    x = x.contiguous();
    weight = weight.contiguous();

    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);  // weight assumed to be (channels, 1, kernel_h, 1)

    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }

    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;

    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    // Divide the batch into tiles for concurrent processing using multiple streams
    int tile_size = (batch + num_streams - 1) / num_streams;

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch kernels asynchronously on different streams over the batch tiles
    for (int tile = 0; tile < batch; tile += tile_size) {
        int current_tile = std::min(tile_size, batch - tile);

        // Offset pointers for the current batch tile
        const float* input_tile = x.data_ptr<float>() + tile * channels * in_h * in_w;
        float* output_tile = output.data_ptr<float>() + tile * channels * out_h * out_w;

        dim3 threads(block_size, 1, 1);
        dim3 blocks((out_w + block_size - 1) / block_size, channels * out_h, current_tile);

        // Select a stream for this tile
        int stream_idx = (tile / tile_size) % num_streams;

        depthwise_conv2d_kernel<<<blocks, threads, 0, streams[stream_idx]>>>(
            input_tile,
            weight.data_ptr<float>(),
            bias_val.data_ptr<float>(),
            output_tile,
            current_tile,
            channels,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            stride,
            padding,
            dilation
        );
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution forward (CUDA) with streams",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"),
          py::arg("num_streams") = 2,
          py::arg("block_size") = 256);
}
