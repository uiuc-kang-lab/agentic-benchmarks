#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define SEG_COUNT 2  // Number of segments to split the input channels

// This kernel splits the input channels into SEG_COUNT segments so that separate blocks can compute partial sums
// for each output element. Each block is responsible for a tile of output pixels for a given batch, output channel,
// and input channel segment. Partial sums are accumulated into global memory using atomicAdd, which is used only
// when combining the segmented results, thereby minimizing contention.

__global__ void conv2d_atomic_kernel(
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

    // Determine output spatial location for this thread
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_row = blockIdx.y * BLOCK_SIZE_Y + ty;
    int out_col = blockIdx.x * BLOCK_SIZE_X + tx;

    // Decode blockIdx.z into segment, output channel, and batch indices
    int temp = blockIdx.z;
    int seg = temp % SEG_COUNT;            // which segment of input channels
    temp /= SEG_COUNT;
    int oc = temp % out_channels;            // output channel index
    int b = temp / out_channels;             // batch index

    if (out_row >= height_out || out_col >= width_out || b >= batch_size)
        return;

    // Calculate the top-left corner in the input for the convolution window
    int in_row_start = out_row * stride - pad_h;
    int in_col_start = out_col * stride - pad_w;

    // Determine the range of input channels for this segment
    int seg_size = (in_channels + SEG_COUNT - 1) / SEG_COUNT;
    int ic_start = seg * seg_size;
    int ic_end = min((seg + 1) * seg_size, in_channels);

    // Initialize sum: add bias only in the first segment
    float sum = (seg == 0 && bias != nullptr) ? bias[oc] : 0.0f;

    // Accumulate partial sum over the designated input channels and the kernel window
    for (int ic = ic_start; ic < ic_end; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            int in_row = in_row_start + kh * dilation_h;
            if (in_row < 0 || in_row >= input_height) continue;
            for (int kw = 0; kw < kernel_w; kw++) {
                int in_col = in_col_start + kw * dilation_w;
                if (in_col < 0 || in_col >= input_width) continue;

                int input_idx = b * in_channels * input_height * input_width +
                                ic * input_height * input_width +
                                in_row * input_width + in_col;
                float x_val = __ldg(&x[input_idx]);

                int weight_idx = oc * in_channels * kernel_h * kernel_w +
                                 ic * kernel_h * kernel_w +
                                 kh * kernel_w + kw;
                sum += x_val * weight[weight_idx];
            }
        }
    }

    // Use atomicAdd to accumulate partial results from different segments into the final output
    int out_idx = b * out_channels * height_out * width_out +
                  oc * height_out * width_out +
                  out_row * width_out + out_col;
    atomicAdd(&output[out_idx], sum);
}


// PyBind11 binding

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
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

    // Initialize output to zeros because atomicAdd accumulates into the output
    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, x.options());

    // Grid configuration:
    //   - blockDim: a tile of output pixels (BLOCK_SIZE_X x BLOCK_SIZE_Y)
    //   - grid.x, grid.y: cover the spatial dimensions of the output
    //   - grid.z: encodes (batch, out_channel, segment) indices
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    int grid_x = (width_out + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    int grid_y = (height_out + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    int grid_z = batch_size * out_channels * SEG_COUNT;
    dim3 blocks(grid_x, grid_y, grid_z);

    conv2d_atomic_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Segmentation-based Conv2D with minimized atomic operations (CUDA)");
}
