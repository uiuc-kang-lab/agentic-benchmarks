#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// This kernel assigns one warp (32 threads) per output pixel. Each warp collaboratively computes the output value for a given (b, out_channel, out_y, out_x) element.
// Each thread in the warp processes a subset of the flattened loop over in_channels * kernelH * kernelW, then a warp-level reduction (using __shfl_down_sync) is performed to combine the partial sums.

__global__ void conv_transpose2d_warp_reduce_kernel(
    const float* __restrict__ input,      // [batch, in_channels, in_height, in_width]
    const float* __restrict__ weight,     // [in_channels, out_channels, kernel_h, kernel_w]
    const float* __restrict__ bias,       // [out_channels] or nullptr
    float* __restrict__ output,           // [batch, out_channels, out_height, out_width]
    int batch,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    // Each block computes one output pixel using exactly 32 threads (one warp).
    // gridDim.x: out_width, gridDim.y: out_height, gridDim.z: batch * out_channels
    int out_x = blockIdx.x;
    int out_y = blockIdx.y;
    int global_oc_b = blockIdx.z; // encodes both output channel and batch index
    int out_channel = global_oc_b % out_channels;
    int b = global_oc_b / out_channels;

    // Ensure blockDim is exactly 32
    int lane = threadIdx.x; // lane id in [0,31]

    // Total iterations: each warp will iterate over in_channels * kernel_h * kernel_w
    int total_iters = in_channels * kernel_h * kernel_w;
    float sum = 0.0f;

    // Loop over the flattened dimension in a strided manner among the 32 threads in a warp
    for (int i = lane; i < total_iters; i += 32) {
        int ic = i / (kernel_h * kernel_w);
        int rem = i % (kernel_h * kernel_w);
        int k_y = rem / kernel_w;
        int k_x = rem % kernel_w;

        // Compute the corresponding input coordinate for this output pixel and kernel position
        int pos_y = out_y + pad_h - k_y;
        int pos_x = out_x + pad_w - k_x;

        // Only contribute if the position aligns with the stride
        if ((pos_y % stride_h == 0) && (pos_x % stride_w == 0)) {
            int in_y = pos_y / stride_h;
            int in_x = pos_x / stride_w;
            // Check if the input coordinate is within bounds
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                int input_idx = ((b * in_channels + ic) * in_height + in_y) * in_width + in_x;
                float input_val = input[input_idx];
                // Weight index: weight has shape [in_channels, out_channels, kernel_h, kernel_w]
                int weight_idx = ((ic * out_channels + out_channel) * kernel_h + k_y) * kernel_w + k_x;
                float weight_val = weight[weight_idx];
                sum += input_val * weight_val;
            }
        }
    }

    // Warp-level reduction using shuffle down intrinsics
    // Full mask for 32 threads
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // The first lane writes the output value
    if (lane == 0) {
        if (bias != nullptr) {
            sum += bias[out_channel];
        }
        int output_idx = ((b * out_channels + out_channel) * out_height + out_y) * out_width + out_x;
        output[output_idx] = sum;
    }
}


// Host function: it sets up the kernel launch with one warp (32 threads) per output element
torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    // Determine bias pointer if bias is provided
    const float* bias_ptr = nullptr;
    torch::Tensor bias;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
        bias_ptr = bias.data_ptr<float>();
    }

    // Input: [batch, in_channels, in_height, in_width]
    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);

    // Weight: [in_channels, out_channels, kernel_h, kernel_w]
    int out_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];

    // Compute output spatial dimensions for transposed convolution
    int out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_width  = (in_width - 1) * stride_w - 2 * pad_w + kernel_w;

    // Allocate output tensor: [batch, out_channels, out_height, out_width]
    auto output = torch::zeros({batch, out_channels, out_height, out_width}, input.options());

    // Each block computes one output pixel using 32 threads (one warp)
    dim3 blockDim(32, 1, 1);
    // Grid dimensions:
    //   x-dim: out_width
    //   y-dim: out_height
    //   z-dim: batch * out_channels
    dim3 gridDim(out_width, out_height, batch * out_channels);

    // Launch the kernel
    conv_transpose2d_warp_reduce_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with warp-level reduction",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
