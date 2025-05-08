#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel distributes the workload evenly by mapping each atomic task
// to a single thread. Each task computes the contribution from one kernel offset
// for a given output element, then uses atomicAdd to accumulate its partial sum.

__global__ void conv_transposed_3d_cuda_kernel_atomic(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int totalAtomicTasks,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int D_out, int H_out, int W_out,
    int groups
) {
    int task = blockIdx.x * blockDim.x + threadIdx.x;
    int Q = kD * kH * kW;  // total number of kernel offset positions
    
    // Precompute per-group channel counts
    int out_channels_per_group = C_out / groups;
    int in_channels_per_group = C_in / groups;

    while (task < totalAtomicTasks) {
        // For each atomic task, decode the flat index:
        int output_index = task / Q;  // index for the output element
        int kernel_offset = task % Q; // which kernel offset (r, s, t)

        // Decode kernel offset into (r, s, t)
        int r = kernel_offset / (kH * kW);
        int rem = kernel_offset % (kH * kW);
        int s = rem / kW;
        int t = rem % kW;

        // Decode output_index into (n, c_out, d, h, w)
        int w = output_index % W_out;
        int tmp = output_index / W_out;
        int h = tmp % H_out;
        tmp /= H_out;
        int d = tmp % D_out;
        tmp /= D_out;
        int c_out = tmp % C_out;
        int n = tmp / C_out;

        // Compute the corresponding input spatial coordinates
        int d_in_calc = d + pad_d - r;
        int h_in_calc = h + pad_h - s;
        int w_in_calc = w + pad_w - t;

        // Check if the computed coordinates align with the stride requirements
        if ((d_in_calc % stride_d) != 0 || (h_in_calc % stride_h) != 0 || (w_in_calc % stride_w) != 0) {
            task += blockDim.x * gridDim.x;
            continue;
        }
        int d_in = d_in_calc / stride_d;
        int h_in = h_in_calc / stride_h;
        int w_in = w_in_calc / stride_w;

        // Check input boundaries
        if (d_in < 0 || d_in >= D_in || h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) {
            task += blockDim.x * gridDim.x;
            continue;
        }

        // Determine the appropriate group for the output channel
        int group = c_out / out_channels_per_group;
        int c_out_in_group = c_out - group * out_channels_per_group;  

        float sum = 0.0f;
        // Accumulate contributions over the input channels belonging to this group
        for (int c = 0; c < in_channels_per_group; c++) {
            int c_in_idx = group * in_channels_per_group + c;
            // Compute input index for (n, c_in_idx, d_in, h_in, w_in)
            int input_index = (((n * C_in + c_in_idx) * D_in + d_in) * H_in + h_in) * W_in + w_in;
            float in_val = input[input_index];

            // Compute weight index. Weight tensor shape: (C_in, C_out_per_group, kD, kH, kW)
            int weight_index = ((c_in_idx * out_channels_per_group + c_out_in_group) * (kD * kH * kW))
                               + (r * kH * kW + s * kW + t);
            float w_val = weight[weight_index];
            
            sum += in_val * w_val;
        }

        // Atomically accumulate the partial sum into the corresponding output element
        atomicAdd(&output[output_index], sum);

        task += blockDim.x * gridDim.x;
    }
}

// Forward function
// Input shape: (N, C_in, D_in, H_in, W_in)
// Weight shape: (C_in, C_out_per_group, kD, kH, kW)
// Bias shape: (C_out) if provided
// Stride, Padding, and Output Padding are 3-element vectors
// Groups: number of groups

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Input dimensions
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    // Kernel dimensions
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    // Stride and padding
    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];
    int out_pad_d = output_padding[0];
    int out_pad_h = output_padding[1];
    int out_pad_w = output_padding[2];

    // Compute output dimensions (assuming dilation = 1)
    int D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + out_pad_d;
    int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + out_pad_h;
    int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + out_pad_w;

    // Calculate output channels
    int out_channels_per_group = weight.size(1);
    int C_out = out_channels_per_group * groups;

    // Initialize the output tensor with bias if provided, else zeros
    torch::Tensor output;
    if (bias.has_value() && bias.value().defined()) {
        output = bias.value().view({1, C_out, 1, 1, 1}).expand({N, C_out, D_out, H_out, W_out}).clone();
    } else {
        output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());
    }

    int totalOutputElements = N * C_out * D_out * H_out * W_out;
    int Q = kD * kH * kW;
    int totalAtomicTasks = totalOutputElements * Q;

    int blockSize = 256;
    int numBlocks = (totalAtomicTasks + blockSize - 1) / blockSize;

    conv_transposed_3d_cuda_kernel_atomic<<<numBlocks, blockSize>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        totalAtomicTasks,
        N, C_in, D_in, H_in, W_in,
        C_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        D_out, H_out, W_out,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward with atomic workload distribution",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
