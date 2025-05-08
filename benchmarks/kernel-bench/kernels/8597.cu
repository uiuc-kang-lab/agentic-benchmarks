#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace py = pybind11;

// Kernel using warp-level primitives for reduction
__global__ void conv_transpose2d_warp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int out_height,
    int out_width
) {
    // Each warp computes one output element
    int total_output = batch_size * out_channels * out_height * out_width;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / warpSize;  // one warp per output element
    int lane = global_tid % warpSize;
    if (warp_id >= total_output) return;

    // Decode warp_id to (n, oc, oh, ow)
    int ow = warp_id % out_width;
    int oh = (warp_id / out_width) % out_height;
    int oc = (warp_id / (out_width * out_height)) % out_channels;
    int n = warp_id / (out_width * out_height * out_channels);

    float sum = 0.0f;
    // Distribute the summation over input channels among the warp lanes
    for (int ic = lane; ic < in_channels; ic += warpSize) {
        // Loop over the kernel spatial dimensions
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Determine the corresponding input coordinate
                int i = oh + padding - kh;
                int j = ow + padding - kw;
                if ((i % stride == 0) && (j % stride == 0)) {
                    i /= stride;
                    j /= stride;
                    if (i >= 0 && i < in_height && j >= 0 && j < in_width) {
                        int input_index = ((n * in_channels + ic) * in_height + i) * in_width + j;
                        int weight_index = ((ic * out_channels + oc) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_index] * weight[weight_index];
                    }
                }
            }
        }
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Lane 0 writes the result
    if (lane == 0) {
        int out_index = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
        output[out_index] = sum;
    }
}

// Forward function which computes output dimensions and launches the kernel

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride_obj = py::int_(1),
    py::object padding_obj = py::int_(0),
    py::object output_padding_obj = py::int_(0),
    int64_t groups = 1
) {
    // x: [N, in_channels, in_height, in_width]
    // weight: [in_channels, out_channels, kernel_size, kernel_size] (square kernel assumed)
    auto x_sizes = x.sizes();
    auto weight_sizes = weight.sizes();
    int batch_size = x_sizes[0];
    int in_channels = x_sizes[1];
    int in_height = x_sizes[2];
    int in_width = x_sizes[3];
    int out_channels = weight_sizes[1];
    int kernel_size = weight_sizes[2];

    int stride = stride_obj.cast<int>();
    int padding = padding_obj.cast<int>();
    int output_padding = output_padding_obj.cast<int>();

    // Compute output dimensions for conv_transpose2d
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    // Each warp computes one output element.
    int total_output_elements = batch_size * out_channels * out_height * out_width;
    // Choose a block size that is a multiple of warpSize (e.g., 128 threads per block => 4 warps per block)
    int threads_per_block = 128;
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (total_output_elements + warps_per_block - 1) / warps_per_block;

    conv_transpose2d_warp_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        out_height,
        out_width
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward with warp-level primitives",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
