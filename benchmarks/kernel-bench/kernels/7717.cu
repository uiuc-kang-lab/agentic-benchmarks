#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use a moderate block size to balance occupancy and work distribution
#define BLOCK_SIZE 256

// This kernel minimizes warp divergence by precomputing the valid convolution ranges
// for depth, height, and width dimensions, so that inner loops execute with uniform control flow
__global__ void conv3d_no_divergence_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int out_depth,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation
) {
    // Determine output channel and batch sample from grid dimensions
    int oc = blockIdx.x;
    int batch_id = blockIdx.y;

    // Total number of output spatial elements
    int total = out_depth * out_height * out_width;

    // Each thread handles multiple output elements
    int tid = threadIdx.x;
    for (int idx = tid; idx < total; idx += blockDim.x) {
        // Map linear index to 3D output coordinates (od, oh, ow)
        int od = idx / (out_height * out_width);
        int rem = idx % (out_height * out_width);
        int oh = rem / out_width;
        int ow = rem % out_width;

        float sum = 0.0f;

        // Compute the top-left-front corner in the input for the convolution window
        int id0 = od * stride - padding;
        int ih0 = oh * stride - padding;
        int iw0 = ow * stride - padding;

        // Precompute valid kernel bounds for depth dimension without inner branch
        int kd_min = (id0 < 0) ? ((-id0 + dilation - 1) / dilation) : 0;
        int kd_max = kernel_d;
        if (id0 + (kernel_d - 1) * dilation >= in_depth) {
            kd_max = (in_depth - id0 + dilation - 1) / dilation;
            if (kd_max > kernel_d) kd_max = kernel_d;
        }

        // Precompute valid kernel bounds for height dimension
        int kh_min = (ih0 < 0) ? ((-ih0 + dilation - 1) / dilation) : 0;
        int kh_max = kernel_h;
        if (ih0 + (kernel_h - 1) * dilation >= in_height) {
            kh_max = (in_height - ih0 + dilation - 1) / dilation;
            if (kh_max > kernel_h) kh_max = kernel_h;
        }

        // Precompute valid kernel bounds for width dimension
        int kw_min = (iw0 < 0) ? ((-iw0 + dilation - 1) / dilation) : 0;
        int kw_max = kernel_w;
        if (iw0 + (kernel_w - 1) * dilation >= in_width) {
            kw_max = (in_width - iw0 + dilation - 1) / dilation;
            if (kw_max > kernel_w) kw_max = kernel_w;
        }

        // Loop over input channels and the computed valid kernel indices
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = kd_min; kd < kd_max; kd++) {
                int id = id0 + kd * dilation; // guaranteed to be in bounds
                for (int kh = kh_min; kh < kh_max; kh++) {
                    int ih = ih0 + kh * dilation; // in bounds
                    for (int kw = kw_min; kw < kw_max; kw++) {
                        int iw = iw0 + kw * dilation; // in bounds

                        // Compute flat indices for input and weight tensors (NCDHW ordering)
                        int input_idx = (((batch_id * in_channels + ic) * in_depth + id) * in_height + ih) * in_width + iw;
                        int weight_idx = (((oc * in_channels + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[oc];
        }

        // Write the result to the output tensor
        int output_idx = (((batch_id * out_channels + oc) * out_depth + od) * out_height + oh) * out_width + ow;
        output[output_idx] = sum;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    TORCH_CHECK(groups == 1, "Only groups=1 is supported");
    auto bias = bias_opt.value_or(at::Tensor());

    // Retrieve input dimensions (N, C, D, H, W)
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Retrieve weight dimensions (out_channels, in_channels, kD, kH, kW)
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Compute output dimensions based on convolution formula
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    at::Tensor output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    // Configure grid: each block processes one output channel of one batch sample
    dim3 grid(out_channels, batch_size);
    conv3d_no_divergence_kernel<<<grid, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D convolution forward with minimized warp divergence (CUDA)");
}
