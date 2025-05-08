#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width,
    const int batch_offset
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = out_channels * out_height * out_width;
    if (idx >= total_elements) return;

    // Compute output position
    const int ow = idx % out_width;
    const int oh = (idx / out_width) % out_height;
    const int oc = (idx / (out_width * out_height));
    const int b = blockIdx.y + batch_offset;
    
    if (b >= batch_size) return;

    const int out_channels_per_group = out_channels / groups;
    const int g = oc / out_channels_per_group;
    const int oc_group = oc % out_channels_per_group;
    const int in_channels_per_group = in_channels / groups;
    const int ic_start = g * in_channels_per_group;

    scalar_t val = (bias != nullptr) ? bias[oc] : scalar_t(0);

    #pragma unroll 4
    for (int kh = 0; kh < kernel_h; ++kh) {
        const int h_in = (oh - kh*dilation + padding) / stride;
        if ((oh - kh*dilation + padding) % stride != 0) continue;
        if (h_in < 0 || h_in >= in_height) continue;

        #pragma unroll 4
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int w_in = (ow - kw*dilation + padding) / stride;
            if ((ow - kw*dilation + padding) % stride != 0) continue;
            if (w_in < 0 || w_in >= in_width) continue;

            #pragma unroll 2
            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                const scalar_t x_val = input[b * in_channels * in_height * in_width +
                                           (ic_start + ic) * in_height * in_width +
                                           h_in * in_width + w_in];
                
                const scalar_t w_val = weight[(ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                            oc_group * kernel_h * kernel_w +
                                            kh * kernel_w + kw];

                val += x_val * w_val;
            }
        }
    }

    output[b * out_channels * out_height * out_width +
           oc * out_height * out_width +
           oh * out_width + ow] = val;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation = 1
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    // Create streams for parallel processing
    const int num_streams = 4;  // Use 4 streams for pipelining
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int batches_per_stream = (batch_size + num_streams - 1) / num_streams;
    const int threads = 512;
    const int blocks_x = (out_channels * out_height * out_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        for (int i = 0; i < num_streams; i++) {
            const int batch_offset = i * batches_per_stream;
            const int batch_size_stream = std::min(batches_per_stream, 
                                                 batch_size - batch_offset);
            if (batch_size_stream <= 0) continue;

            dim3 grid(blocks_x, batch_size_stream);
            conv_transpose2d_kernel<scalar_t><<<grid, threads, 0, streams[i]>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                (bias.has_value() && bias->defined()) ? bias->data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                batch_size,
                in_channels,
                in_height,
                in_width,
                out_channels,
                kernel_h,
                kernel_w,
                stride,
                padding,
                groups,
                dilation,
                out_height,
                out_width,
                batch_offset
            );
        }
    }));

    // Synchronize and cleanup streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 2D convolution (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}