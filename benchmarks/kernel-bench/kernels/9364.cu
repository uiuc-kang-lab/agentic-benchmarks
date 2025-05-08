#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses warp-level primitives to compute one output element per block.
// Each block (of 32 threads) computes a single output pixel for a given (batch, out_channel, h, w).
// The work of computing the convolution sum is split among the 32 threads, and the partial sums are reduced
// using __shfl_down_sync. This eliminates the need for shared memory reductions, thereby reducing synchronization overhead.

__global__ void conv2d_warp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int height_out,
    int width_out,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {

    // Each block computes one output element
    int out_elem = blockIdx.x; // one block per output element
    int total_outputs = batch_size * out_channels * height_out * width_out;
    if (out_elem >= total_outputs) return;

    // Decode the linear index into (b, oc, h_out, w_out)
    int w_out = out_elem % width_out;
    int tmp = out_elem / width_out;
    int h_out = tmp % height_out;
    tmp = tmp / height_out;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    // Each block has 32 threads (one warp). Get the lane id (0..31).
    int lane = threadIdx.x; // blockDim.x should be 32.

    // Total number of iterations over (ic, kh, kw)
    int total_iter = in_channels * kernel_h * kernel_w;

    // Initialize with bias if provided
    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // Each thread loops over a strided range of the summation
    for (int iter = lane; iter < total_iter; iter += 32) {
        int ic = iter / (kernel_h * kernel_w);
        int rem = iter % (kernel_h * kernel_w);
        int kh = rem / kernel_w;
        int kw = rem % kernel_w;

        int h_in = h_out * stride + kh * dilation_h - pad_h;
        int w_in = w_out * stride + kw * dilation_w - pad_w;
        float x_val = 0.0f;
        if ((h_in >= 0) && (h_in < input_height) && (w_in >= 0) && (w_in < input_width)) {
            int x_idx = b * (in_channels * input_height * input_width) +
                        ic * (input_height * input_width) +
                        h_in * input_width + w_in;
            x_val = __ldg(&x[x_idx]);
        }
        
        int weight_idx = oc * (in_channels * kernel_h * kernel_w) +
                         ic * (kernel_h * kernel_w) +
                         kh * kernel_w + kw;
        float w_val = __ldg(&weight[weight_idx]);
        
        sum += x_val * w_val;
    }

    // Warp-level reduction using __shfl_down_sync to sum across lanes
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Lane 0 writes the final result
    if (lane == 0) {
        int out_idx = b * (out_channels * height_out * width_out) +
                      oc * (height_out * width_out) +
                      h_out * width_out + w_out;
        output[out_idx] = sum;
    }
}

// PyBind11 binding

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,  // optional bias
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        bias_ptr = bias->data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);

    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    // Total number of output elements
    int total_outputs = batch_size * out_channels * height_out * width_out;
    // Launch one block per output element with 32 threads per block (one warp)
    int threads = 32;
    int blocks = total_outputs;

    conv2d_warp_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        height_out,
        width_out,
        stride,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Conv2D forward (CUDA) with warp-level reduction");
}
