#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Kernel that performs the transposed 2D convolution for a batch chunk
// It assumes that the input pointer points to a buffer containing "batch_size" images
// and writes to an output buffer of corresponding size.

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    const int total_elements = batch_size * out_channels * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int n = idx;
    const int ow = n % out_width;
    n /= out_width;
    const int oh = n % out_height;
    n /= out_height;
    const int oc = n % out_channels;
    n /= out_channels;
    const int b = n;  // local batch index within the chunk

    // Initialize output value using bias if provided
    scalar_t val = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

    const int out_channels_per_group = out_channels / groups;
    const int g = oc / out_channels_per_group;
    const int oc_group = oc % out_channels_per_group;
    const int in_channels_per_group = in_channels / groups;
    const int ic_start = g * in_channels_per_group;

    // Iterate over the kernel window
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int h_in = (oh - kh * dilation + padding) / stride;
            if ((oh - kh * dilation + padding) % stride != 0) continue;
            if (h_in < 0 || h_in >= in_height) continue;

            const int w_in = (ow - kw * dilation + padding) / stride;
            if ((ow - kw * dilation + padding) % stride != 0) continue;
            if (w_in < 0 || w_in >= in_width) continue;

            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                const scalar_t x_val = input[b * in_channels * in_height * in_width +
                                               (ic_start + ic) * in_height * in_width +
                                               h_in * in_width +
                                               w_in];
                const scalar_t w_val = weight[(ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                              oc_group * (kernel_h * kernel_w) +
                                              kh * kernel_w +
                                              kw];
                val += x_val * w_val;
            }
        }
    }
    output[idx] = val;
}

// Helper function to launch the kernel for a given chunk using the provided CUDA stream

template <typename scalar_t>
void launch_chunk_kernel(
    const scalar_t* d_input,
    const scalar_t* d_weight,
    const scalar_t* d_bias,
    scalar_t* d_output,
    int chunk_batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int out_height,
    int out_width,
    cudaStream_t stream
) {
    const int total_elements = chunk_batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    conv_transpose2d_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        d_input,
        d_weight,
        d_bias,
        d_output,
        chunk_batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_h,
        kernel_w,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        out_height,
        out_width
    );
}

// The forward function splits the input batch into chunks and overlaps asynchronous device-to-device
// memory copies (input and output transfers) with kernel execution using double buffering and CUDA streams.

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

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == out_channels, "Bias must have out_channels elements");
        TORCH_CHECK(bias->device().is_cuda(), "Bias must be a CUDA tensor");
    }

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    // Number of elements per image in the input and output
    const int input_batch_elems = in_channels * in_height * in_width;
    const int output_batch_elems = out_channels * out_height * out_width;

    // Determine chunk size for pipelining (tunable parameter)
    int chunk_size = batch_size > 16 ? 16 : batch_size;
    int num_chunks = (batch_size + chunk_size - 1) / chunk_size;

    // Create two CUDA streams for double buffering
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda_async", ([&] {
        using scalar_t_ = scalar_t;
        size_t in_chunk_bytes = input_batch_elems * sizeof(scalar_t_);
        size_t out_chunk_bytes = output_batch_elems * sizeof(scalar_t_);

        // Allocate double buffers for input and output chunks on device
        scalar_t_* d_temp_input[2];
        scalar_t_* d_temp_output[2];
        cudaMalloc(&d_temp_input[0], chunk_size * in_chunk_bytes);
        cudaMalloc(&d_temp_input[1], chunk_size * in_chunk_bytes);
        cudaMalloc(&d_temp_output[0], chunk_size * out_chunk_bytes);
        cudaMalloc(&d_temp_output[1], chunk_size * out_chunk_bytes);

        for (int i = 0; i < num_chunks; i++) {
            int current_chunk = std::min(chunk_size, batch_size - i * chunk_size);
            int stream_idx = i % 2;
            cudaStream_t stream = streams[stream_idx];

            // Asynchronously copy the input chunk from the global input tensor to the temporary buffer
            const scalar_t_* src_ptr = reinterpret_cast<const scalar_t_*>(x.data_ptr<scalar_t_>()) + i * chunk_size * input_batch_elems;
            cudaMemcpyAsync(d_temp_input[stream_idx], src_ptr,
                            current_chunk * in_chunk_bytes,
                            cudaMemcpyDeviceToDevice, stream);

            // Launch the convolution kernel on the temporary input buffer
            launch_chunk_kernel<scalar_t_>(
                d_temp_input[stream_idx],
                weight.data_ptr<scalar_t_>(),
                (bias.has_value() && bias->defined()) ? bias->data_ptr<scalar_t_>() : nullptr,
                d_temp_output[stream_idx],
                current_chunk,
                in_channels,
                in_height,
                in_width,
                out_channels,
                kernel_h,
                kernel_w,
                stride,
                padding,
                output_padding,
                groups,
                dilation,
                out_height,
                out_width,
                stream
            );

            // Asynchronously copy the resulting output chunk to the final output tensor
            scalar_t_* dst_ptr = reinterpret_cast<scalar_t_*>(output.data_ptr<scalar_t_>()) + i * chunk_size * output_batch_elems;
            cudaMemcpyAsync(dst_ptr, d_temp_output[stream_idx],
                            current_chunk * out_chunk_bytes,
                            cudaMemcpyDeviceToDevice, stream);
        }

        // Synchronize both streams to ensure all operations are complete
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);

        // Free temporary buffers
        cudaFree(d_temp_input[0]);
        cudaFree(d_temp_input[1]);
        cudaFree(d_temp_output[0]);
        cudaFree(d_temp_output[1]);
    }));

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 2D convolution with async pipelining (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
