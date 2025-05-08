#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D transposed convolution
// This kernel assigns one output element per thread, computing its value
// by gathering contributions from the input and weight tensors. Workload
// is evenly distributed across threads and blocks.

// Assumptions:
// - Input tensor shape: [N, in_channels, in_d, in_h, in_w]
// - Weight tensor shape: [in_channels_per_group, out_channels_per_group, kernel_d, kernel_h, kernel_w]
//   where in_channels_per_group = in_channels / groups and out_channels_per_group = out_channels / groups
// - Bias tensor (if provided) has shape: [out_channels]
// - Output tensor shape is computed as:
//    out_d = (in_d - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d
//    out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h
//    out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w

__global__ void transposed_conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr if not provided
    float* __restrict__ output,
    int N,
    int in_channels,
    int in_d, int in_h, int in_w,
    int out_channels,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int groups
) {
    // Compute total number of output elements
    int total = N * out_channels * out_d * out_h * out_w;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;

    // Decode the flattened index into n, oc, od, oh, ow
    int out_spatial = out_d * out_h * out_w; 
    int n = index / (out_channels * out_spatial);
    int rem = index % (out_channels * out_spatial);
    int oc = rem / out_spatial;
    rem = rem % out_spatial;
    int od = rem / (out_h * out_w);
    rem = rem % (out_h * out_w);
    int oh = rem / out_w;
    int ow = rem % out_w;

    // Determine group-related parameters
    int out_channels_per_group = out_channels / groups;  // weight dimension 1
    int in_channels_per_group = in_channels / groups;
    int group = oc / out_channels_per_group;
    int oc_local = oc % out_channels_per_group;

    // Initialize accumulator with bias if provided
    float sum = 0.0f;
    if (bias != nullptr) {
        sum = bias[oc];
    }

    // Iterate over the kernel dimensions, and for valid positions, accumulate contributions
    for (int kd = 0; kd < kernel_d; kd++) {
        int d_temp = od + pad_d - kd;  // expected to equal id * stride_d
        if (d_temp < 0 || d_temp % stride_d != 0) continue;
        int id = d_temp / stride_d;
        if (id < 0 || id >= in_d) continue;

        for (int kh = 0; kh < kernel_h; kh++) {
            int h_temp = oh + pad_h - kh;
            if (h_temp < 0 || h_temp % stride_h != 0) continue;
            int ih = h_temp / stride_h;
            if (ih < 0 || ih >= in_h) continue;

            for (int kw = 0; kw < kernel_w; kw++) {
                int w_temp = ow + pad_w - kw;
                if (w_temp < 0 || w_temp % stride_w != 0) continue;
                int iw = w_temp / stride_w;
                if (iw < 0 || iw >= in_w) continue;

                // Sum over the input channels within the current group
                for (int c = 0; c < in_channels_per_group; c++) {
                    int c_in = group * in_channels_per_group + c;
                    // Compute linear index for input: [n, c_in, id, ih, iw]
                    int input_index = (((n * in_channels + c_in) * in_d + id) * in_h + ih) * in_w + iw;

                    // Compute linear index for weight: [c, oc_local, kd, kh, kw]
                    int weight_index = ((((c) * out_channels_per_group + oc_local) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;

                    sum += input[input_index] * weight[weight_index];
                } // end for c
            } // end for kw
        } // end for kh
    } // end for kd

    // Write the computed sum to the output tensor at [n, oc, od, oh, ow]
    int output_index = (((n * out_channels + oc) * out_d + od) * out_h + oh) * out_w + ow;
    output[output_index] = sum;
}

// Launcher function for the CUDA kernel
// This function computes the output dimensions, sets up the grid and block sizes, and calls the kernel

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Check inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
    }

    // Get input dimensions
    int N = x.size(0);
    int in_channels = x.size(1);
    int in_d = x.size(2);
    int in_h = x.size(3);
    int in_w = x.size(4);

    // Get kernel dimensions
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Stride, padding, and output padding
    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];
    int out_pad_d = output_padding[0];
    int out_pad_h = output_padding[1];
    int out_pad_w = output_padding[2];

    // Compute output spatial dimensions
    int out_d = (in_d - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;

    // Compute output channels: weight shape is [in_channels, out_channels_per_group, kernel_d, kernel_h, kernel_w]
    int out_channels_per_group = weight.size(1);
    int out_channels = groups * out_channels_per_group;

    // Create output tensor
    auto output = torch::zeros({N, out_channels, out_d, out_h, out_w}, x.options());

    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    // Total number of output elements
    int total_elements = N * out_channels * out_d * out_h * out_w;

    // Launch the kernel with an appropriate block and grid size
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    
    transposed_conv3d_kernel<<<numBlocks, blockSize>>>(
        x_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        N,
        in_channels,
        in_d, in_h, in_w,
        out_channels,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        groups
    );

    // Ensure kernel launch success
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

// PyBind11 module registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "3D Transposed Convolution forward (CUDA) with even workload distribution");
}
