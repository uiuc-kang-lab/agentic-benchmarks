#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using 3D grid mapping: gridDim.z covers batch_size*out_channels,
// gridDim.x and gridDim.y cover spatial dimensions output_w and output_h respectively.

__global__ void depthwise_3dgrid_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group
) {
    // Determine the (batch, channel) index from gridDim.z
    int oc_batch = blockIdx.z;              // combined index for (batch, channel)
    int b = oc_batch / out_channels;
    int oc = oc_batch % out_channels;

    // Determine spatial coordinates for the output
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;

    if (w_out >= output_w || h_out >= output_h) return;

    // Map oc to input channel and weight index
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    /* Load input tile into shared memory */
    extern __shared__ float s_input[];
    int shared_w = blockDim.x * stride + kernel_size - stride;
    int shared_h = blockDim.y * stride + kernel_size - stride;
    int out_w_start = blockIdx.x * blockDim.x;
    int out_h_start = blockIdx.y * blockDim.y;
    int in_w_start = out_w_start * stride - padding;
    int in_h_start = out_h_start * stride - padding;
    for (int i = threadIdx.y; i < shared_h; i += blockDim.y) {
        for (int j = threadIdx.x; j < shared_w; j += blockDim.x) {
            int in_w = in_w_start + j;
            int in_h = in_h_start + i;
            float inp = 0.0f;
            if (in_w >= 0 && in_w < input_w && in_h >= 0 && in_h < input_h) {
                int input_idx = b * (in_channels * input_h * input_w) +
                                in_ch * (input_h * input_w) +
                                in_h * input_w + in_w;
                inp = input[input_idx];
            }
            s_input[i * shared_w + j] = inp;
        }
    }
    __syncthreads();

    /* Compute convolution using data from shared memory */
    float sum = 0.0f;
    int local_x = threadIdx.x * stride;
    int local_y = threadIdx.y * stride;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int s_index = (local_y + kh) * shared_w + (local_x + kw);
            int weight_idx = in_ch * (channels_per_group * kernel_size * kernel_size) +
                             weight_ch * (kernel_size * kernel_size) +
                             kh * kernel_size + kw;
            sum += s_input[s_index] * weight[weight_idx];
        }
    }
        int h_in = h_out * stride + kh - padding;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w_in = w_out * stride + kw - padding;
            if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                int input_idx = b * (in_channels * input_h * input_w) +
                                in_ch * (input_h * input_w) +
                                h_in * input_w + w_in;
                int weight_idx = in_ch * (channels_per_group * kernel_size * kernel_size) +
                                 weight_ch * (kernel_size * kernel_size) +
                                 kh * kernel_size + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    int out_idx = b * (out_channels * output_h * output_w) +
                  oc * (output_h * output_w) +
                  h_out * output_w + w_out;
    output[out_idx] = sum;
}


torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Set up block and grid dimensions.
    // Use a 2D block for spatial dimensions (e.g. 16x16 threads)
    const int TILE_W = 16;
    const int TILE_H = 16;
    dim3 blockDim(TILE_W, TILE_H, 1);
    
    // Grid dimensions: x covers output_w, y covers output_h, and z covers batch * out_channels
    dim3 gridDim((output_w + TILE_W - 1) / TILE_W,
                 (output_h + TILE_H - 1) / TILE_H,
                 batch_size * out_channels);

    depthwise_3dgrid_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_h,
        input_w,
        out_channels,
        output_h,
        output_w,
        kernel_size,
        stride,
        padding,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution with optimized 3D grid mapping",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
