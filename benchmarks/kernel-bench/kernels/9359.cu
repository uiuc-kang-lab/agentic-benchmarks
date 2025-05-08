#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Each warp computes one output element via warp-level reduction
__global__ void conv2d_kernel(
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

    // Each warp (32 threads) computes one output element
    // blockDim.x must be WARP_SIZE (32), and blockDim.y gives number of warps per block
    int warp_id_in_block = threadIdx.y;  // which warp in the block
    int lane = threadIdx.x;              // lane index within the warp

    // Global warp index
    int global_warp_id = blockIdx.x * blockDim.y + warp_id_in_block;

    // Total number of output elements
    int total_output = batch_size * out_channels * height_out * width_out;
    if (global_warp_id >= total_output) return;

    // Map global warp id to output indices
    int tmp = global_warp_id;
    int w_out = tmp % width_out;
    tmp /= width_out;
    int h_out = tmp % height_out;
    tmp /= height_out;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float thread_sum = 0.0f;

    // Each thread in the warp processes a subset of input channels
    // using a stride of WARP_SIZE over the in_channels dimension
    for (int ic = lane; ic < in_channels; ic += WARP_SIZE) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = h_out * stride + kh * dilation_h - pad_h;
            if (h_in < 0 || h_in >= input_height) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_in = w_out * stride + kw * dilation_w - pad_w;
                if (w_in < 0 || w_in >= input_width) continue;

                int x_index = b * in_channels * input_height * input_width +
                              ic * input_height * input_width +
                              h_in * input_width + w_in;
                float x_val = __ldg(&x[x_index]);

                int weight_index = oc * in_channels * kernel_h * kernel_w +
                                   ic * kernel_h * kernel_w +
                                   kh * kernel_w + kw;
                float w_val = weight[weight_index];

                thread_sum += x_val * w_val;
            }
        }
    }

    // Warp-level reduction using __shfl_down_sync to sum partial results
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // The first lane of the warp writes the output
    if (lane == 0) {
        if (bias != nullptr) {
            thread_sum += bias[oc];
        }
        int out_index = b * out_channels * height_out * width_out +
                        oc * height_out * width_out +
                        h_out * width_out + w_out;
        output[out_index] = thread_sum;
    }
}


// Forward function callable from PyTorch
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias, // optional bias
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
    int total_output = batch_size * out_channels * height_out * width_out;

    // Each warp computes one output element.
    // Choose block dimensions: 32 threads in x (one warp) and, say, 8 warps per block.
    dim3 threads(WARP_SIZE, 8, 1);
    int warps_per_block = threads.y;
    int num_blocks = (total_output + warps_per_block - 1) / warps_per_block;
    dim3 blocks(num_blocks, 1, 1);

    conv2d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Conv2D forward (CUDA)");
}
