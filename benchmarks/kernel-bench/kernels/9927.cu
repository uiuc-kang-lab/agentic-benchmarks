#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int KERNEL_SIZE>
__global__ void depthwise_3dgrid_unrolled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int input_h,
    const int input_w,
    const int out_channels,
    const int output_h,
    const int output_w,
    const int stride,
    const int padding,
    const int channels_per_group
) {
    const int oc_batch = blockIdx.z;
    const int b = oc_batch / out_channels;
    const int oc = oc_batch % out_channels;
    
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (w_out >= output_w || h_out >= output_h) return;

    const int in_ch = oc / channels_per_group;
    const int weight_ch = oc % channels_per_group;

    // Precompute base indices
    const int input_batch_offset = b * (in_channels * input_h * input_w);
    const int input_channel_offset = in_ch * (input_h * input_w);
    const int weight_offset = in_ch * (channels_per_group * KERNEL_SIZE * KERNEL_SIZE) +
                             weight_ch * (KERNEL_SIZE * KERNEL_SIZE);

    float sum = 0.0f;

    // Manual unroll for KERNEL_SIZE x KERNEL_SIZE
    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
        const int h_in = h_out * stride + kh - padding;
        
        if (h_in >= 0 && h_in < input_h) {
            const int input_h_offset = input_batch_offset + input_channel_offset + h_in * input_w;
            const int weight_h_offset = weight_offset + kh * KERNEL_SIZE;
            
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                const int w_in = w_out * stride + kw - padding;
                
                if (w_in >= 0 && w_in < input_w) {
                    sum += __ldg(&input[input_h_offset + w_in]) * 
                           __ldg(&weight[weight_h_offset + kw]);
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += __ldg(&bias[oc]);
    }

    output[b * (out_channels * output_h * output_w) +
           oc * (output_h * output_w) +
           h_out * output_w + 
           w_out] = sum;
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
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int kernel_size = weight.size(2);
    const int channels_per_group = weight.size(1);
    const int out_channels = in_channels * channels_per_group;

    const int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Optimize block dimensions for H100
    const int TILE_W = 16;
    const int TILE_H = 16;
    dim3 block(TILE_W, TILE_H);
    dim3 grid((output_w + TILE_W - 1) / TILE_W,
              (output_h + TILE_H - 1) / TILE_H,
              batch_size * out_channels);

    // Template specialization for common kernel sizes
    if (kernel_size == 3) {
        depthwise_3dgrid_unrolled_kernel<3><<<grid, block>>>(
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
            stride,
            padding,
            channels_per_group
        );
    } else if (kernel_size == 5) {
        depthwise_3dgrid_unrolled_kernel<5><<<grid, block>>>(
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
            stride,
            padding,
            channels_per_group
        );
    } else {
        // Fallback for other kernel sizes
        depthwise_3dgrid_unrolled_kernel<7><<<grid, block>>>(
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
            stride,
            padding,
            channels_per_group
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution with unrolled 3D grid mapping",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}