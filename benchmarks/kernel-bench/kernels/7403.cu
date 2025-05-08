#include <torch/extension.h>

// Forward function definition
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // Ensure inputs are on CUDA and contiguous
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    // Get tensor dimensions
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto in_height = x.size(2);
    auto in_width = x.size(3);
    auto out_channels = weight.size(1);
    auto kernel_size = weight.size(2);

    // Calculate output dimensions
    auto out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width},
                              x.options());

    // Get raw pointers for direct memory access
    const float* __restrict__ x_ptr = x.data_ptr<float>();
    const float* __restrict__ weight_ptr = weight.data_ptr<float>();
    float* __restrict__ output_ptr = output.data_ptr<float>();
    const float* __restrict__ bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    // Use at::parallel_for to parallelize the computation
    at::parallel_for(0, batch_size * out_channels, 0, [&](int64_t begin, int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
            int64_t n = index / out_channels;
            int64_t oc = index % out_channels;

            // Use __ldg for read-only memory access
            float bias_val = bias_ptr ? __ldg(&bias_ptr[oc]) : 0.0f;

            for (int64_t ic = 0; ic < in_channels; ++ic) {
                for (int64_t ih = 0; ih < in_height; ++ih) {
                    for (int64_t iw = 0; iw < in_width; ++iw) {
                        // Calculate input offset for aligned access
                        int64_t input_offset = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                        float input_val = __ldg(&x_ptr[input_offset]);

                        for (int64_t kh = 0; kh < kernel_size; ++kh) {
                            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                                int64_t oh = ih * stride - padding + kh;
                                int64_t ow = iw * stride - padding + kw;

                                if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                                    // Calculate weight offset for aligned access
                                    int64_t weight_offset = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                    float weight_val = __ldg(&weight_ptr[weight_offset]);

                                    // Calculate output offset for aligned access
                                    int64_t output_offset = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
                                    output_ptr[output_offset] += input_val * weight_val;
                                }
                            }
                        }
                    }
                }
            }

            // Add bias if present
            if (bias_ptr) {
                for (int64_t oh = 0; oh < out_height; ++oh) {
                    for (int64_t ow = 0; ow < out_width; ++ow) {
                        int64_t output_offset = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
                        output_ptr[output_offset] += bias_val;
                    }
                }
            }
        }
    });

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA)");
}