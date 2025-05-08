#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions chosen to match warp/coalescing requirements
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Coalesced Conv2D Kernel
// Each block computes a TILE_HEIGHT x TILE_WIDTH patch for a single output channel of one batch element.
// This design ensures that threads in a warp access contiguous memory locations when reading input x and writing to output.
__global__ void conv2d_kernel(
    const float* __restrict__ x,        // [batch, in_channels, input_height, input_width]
    const float* __restrict__ weight,   // [out_channels, in_channels, kernel_h, kernel_w]
    const float* __restrict__ bias,     // [out_channels] (optional)
    float* __restrict__ output,         // [batch, out_channels, height_out, width_out]
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
    int dilation_w) 
{
    // Compute output spatial location for this thread
    int out_col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int out_row = blockIdx.y * TILE_HEIGHT + threadIdx.y;

    // Each block is mapped to a specific (batch, output channel) pair
    int bc = blockIdx.z;  // bc = b * out_channels + oc
    int b  = bc / out_channels;
    int oc = bc % out_channels;

    if (out_row < height_out && out_col < width_out) {
        // Initialize accumulator with bias if provided
        float acc = (bias != nullptr) ? bias[oc] : 0.0f;

        // Iterate over all input channels
        for (int ic = 0; ic < in_channels; ic++) {
            // For each kernel element
            for (int kh = 0; kh < kernel_h; kh++) {
                int in_row = out_row * stride + kh * dilation_h - pad_h;
                // Check row validity once per kernel row
                if (in_row < 0 || in_row >= input_height) continue;
                for (int kw = 0; kw < kernel_w; kw++) {
                    int in_col = out_col * stride + kw * dilation_w - pad_w;
                    if (in_col < 0 || in_col >= input_width) continue;

                    // Compute the index for input x
                    int x_idx = b * in_channels * input_height * input_width +
                                ic * input_height * input_width +
                                in_row * input_width + in_col;

                    // Compute the index for weight for current output channel oc
                    int w_idx = oc * in_channels * kernel_h * kernel_w +
                                ic * kernel_h * kernel_w +
                                kh * kernel_w + kw;

                    acc += __ldg(&x[x_idx]) * __ldg(&weight[w_idx]);
                }
            }
        }

        // Write output ensuring coalesced access: threads in the same block write contiguous locations
        int out_idx = b * out_channels * height_out * width_out +
                      oc * height_out * width_out +
                      out_row * width_out + out_col;
        output[out_idx] = acc;
    }
}


// PyBind11 forward function
// This function validates tensors, computes output dimensions, and launches the kernel with coalesced memory accesses

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,  // optional bias
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation) 
{
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
    int input_width  = x.size(3);

    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out  = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    // Define grid and block dimensions
    dim3 threads(TILE_WIDTH, TILE_HEIGHT);
    int grid_x = (width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (height_out + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int grid_z = batch_size * out_channels;  // one block per (batch, output channel) pair
    dim3 blocks(grid_x, grid_y, grid_z);

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
    m.def("forward", &forward, "Coalesced Conv2D forward (CUDA)");
}
