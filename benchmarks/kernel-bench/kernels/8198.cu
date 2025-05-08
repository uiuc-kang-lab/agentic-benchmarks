#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Optimized kernel with __restrict__ pointers and loop unrolling
template <typename scalar_t>
__global__ void conv_transpose2d_async_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,    // chunk batch size
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * out_height * out_width;
    if (idx >= total_elements) return;

    // Compute output coordinates
    int ow = idx % out_width;
    int n = idx / out_width;
    int oh = n % out_height;
    n /= out_height;
    int oc = n % out_channels;
    int b = n / out_channels;

    // Determine group parameters
    const int out_channels_per_group = out_channels / groups;
    const int g = oc / out_channels_per_group;
    const int oc_group = oc % out_channels_per_group;
    const int in_channels_per_group = in_channels / groups;
    const int ic_start = g * in_channels_per_group;

    const int input_batch_stride = in_channels * in_height * in_width;
    const int weight_channel_stride = kernel_h * kernel_w;

    // Initialize with bias if provided
    scalar_t val = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        int h_in_base = oh - kh * dilation + padding;
        if (h_in_base % stride != 0) continue;
        int h_in = h_in_base / stride;
        if (h_in < 0 || h_in >= in_height) continue;

        #pragma unroll
        for (int kw = 0; kw < kernel_w; ++kw) {
            int w_in_base = ow - kw * dilation + padding;
            if (w_in_base % stride != 0) continue;
            int w_in = w_in_base / stride;
            if (w_in < 0 || w_in >= in_width) continue;

            int in_spatial_offset = h_in * in_width + w_in;
            int weight_offset = oc_group * weight_channel_stride + kh * kernel_w + kw;
            
            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                int input_idx = b * input_batch_stride + (ic_start + ic) * (in_height * in_width) + in_spatial_offset;
                int weight_idx = (ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w) + weight_offset;
                val += input[input_idx] * weight[weight_idx];
            }
        }
    }
    output[idx] = val;
}

// Kernel launcher for a chunk using a specified CUDA stream
template <typename scalar_t>
void launch_async_kernel(
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
    // Use thread block size that's a multiple of warp size (32) and fits SM constraints
    const int thread_block_size = 256;  // 8 warps per block
    // Calculate 2D grid to better match the spatial nature of convolution
    dim3 block(thread_block_size);
    // Use 2D grid to better map to output dimensions
    int grid_x = (out_width + block.x - 1) / block.x;
    int grid_y = (total_elements/out_width + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    
    conv_transpose2d_async_kernel<scalar_t><<<grid, block, 0, stream>>>(
        d_input, d_weight, d_bias, d_output,
        chunk_batch_size, in_channels, in_height, in_width,
        out_channels, kernel_h, kernel_w, stride, padding, output_padding,
        groups, dilation, out_height, out_width
    );
}

// Forward function with asynchronous, double buffered pipelining
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

    // Compute output dimensions
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    // Determine chunking parameters for pipelining
    const int default_chunk_size = 16;
    int chunk_size = (batch_size >= default_chunk_size) ? default_chunk_size : batch_size;
    int num_chunks = (batch_size + chunk_size - 1) / chunk_size;

    const int input_batch_elems = in_channels * in_height * in_width;
    const int output_batch_elems = out_channels * out_height * out_width;

    // Create two CUDA streams for double buffering
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_async_forward", ([&] {
        using scalar_t = scalar_t;
        size_t in_chunk_bytes = chunk_size * input_batch_elems * sizeof(scalar_t);
        size_t out_chunk_bytes = chunk_size * output_batch_elems * sizeof(scalar_t);

        // Allocate double buffers for input and output chunks on-device
        scalar_t* d_temp_input[2];
        scalar_t* d_temp_output[2];
        cudaMalloc(&d_temp_input[0], in_chunk_bytes);
        cudaMalloc(&d_temp_input[1], in_chunk_bytes);
        cudaMalloc(&d_temp_output[0], out_chunk_bytes);
        cudaMalloc(&d_temp_output[1], out_chunk_bytes);

        for (int i = 0; i < num_chunks; i++) {
            int current_chunk = std::min(chunk_size, batch_size - i * chunk_size);
            int stream_idx = i % 2;
            cudaStream_t stream = streams[stream_idx];

            // Asynchronously copy the input chunk (device-to-device)
            const scalar_t* src_ptr = x.data_ptr<scalar_t>() + i * chunk_size * input_batch_elems;
            cudaMemcpyAsync(d_temp_input[stream_idx], src_ptr,
                            current_chunk * input_batch_elems * sizeof(scalar_t),
                            cudaMemcpyDeviceToDevice, stream);

            // Launch the optimized convolution kernel on the current chunk
            launch_async_kernel<scalar_t>(
                d_temp_input[stream_idx],
                weight.data_ptr<scalar_t>(),
                (bias.has_value() && bias->defined()) ? bias->data_ptr<scalar_t>() : nullptr,
                d_temp_output[stream_idx],
                current_chunk,
                in_channels, in_height, in_width,
                out_channels, kernel_h, kernel_w,
                stride, padding, output_padding,
                groups, dilation,
                out_height, out_width,
                stream
            );

            // Asynchronously copy the resulting output chunk to the final output tensor
            scalar_t* dst_ptr = output.data_ptr<scalar_t>() + i * chunk_size * output_batch_elems;
            cudaMemcpyAsync(dst_ptr, d_temp_output[stream_idx],
                            current_chunk * output_batch_elems * sizeof(scalar_t),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // Synchronize streams
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);

        // Free temporary double-buffered memory
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
    m.def("forward", &forward,
          "Optimized Transposed 2D Convolution with async pipelining and double buffering (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
