#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void depthwise_conv2d_kernel(
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
    __shared__ float input_tile[TILE_SIZE + 2][TILE_SIZE + 2]; // +2 for kernel_size=3

    int b = blockIdx.z;
    int in_ch = blockIdx.y;
    int h_tile = (blockIdx.x * TILE_SIZE) / output_w;
    int w_tile = (blockIdx.x * TILE_SIZE) % output_w;

    int h_out = h_tile + threadIdx.y;
    int w_out = w_tile + threadIdx.x;
    if (h_out >= output_h || w_out >= output_w) return;

    // Load input tile into shared memory
    int h_in_base = h_out * stride - padding;
    int w_in_base = w_out * stride - padding;
    
    for (int kh = 0; kh < kernel_size; kh += blockDim.y) {
        for (int kw = 0; kw < kernel_size; kw += blockDim.x) {
            int h_in = h_in_base + kh + threadIdx.y;
            int w_in = w_in_base + kw + threadIdx.x;
            
            if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                input_tile[threadIdx.y + kh][threadIdx.x + kw] = 
                    input[b * in_channels * input_h * input_w + 
                         in_ch * input_h * input_w + 
                         h_in * input_w + w_in];
            } else {
                input_tile[threadIdx.y + kh][threadIdx.x + kw] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Process all channels in group
    for (int ch = 0; ch < channels_per_group; ch++) {
        float sum = 0.0f;
        int oc = in_ch * channels_per_group + ch;
        
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                sum += input_tile[threadIdx.y * stride + kh][threadIdx.x * stride + kw] *
                       weight[in_ch * channels_per_group * kernel_size * kernel_size +
                              ch * kernel_size * kernel_size +
                              kh * kernel_size + kw];
            }
        }
        
        if (bias) sum += bias[oc];
        
        if (oc < out_channels) {
            output[b * out_channels * output_h * output_w +
                  oc * output_h * output_w +
                  h_out * output_w + w_out] = sum;
        }
    }
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
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    int num_tiles = ((output_h + TILE_SIZE - 1) / TILE_SIZE) * ((output_w + TILE_SIZE - 1) / TILE_SIZE);
    dim3 blocks(num_tiles, in_channels, batch_size);

    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
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
    m.def("forward", &forward, "Depthwise 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
