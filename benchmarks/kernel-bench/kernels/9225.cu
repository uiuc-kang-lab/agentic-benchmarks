#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel assigns one warp (32 threads) to compute each output element.
// Each warp's threads split the work of summing over (in_channels * kernel_size) iterations
// using warp-level reduction with __shfl_down_sync. This avoids global atomic operations,
// as the threads within a warp combine their partial contributions without race conditions.

__global__ void conv_transpose1d_warp_kernel(
    const float* __restrict__ x_ptr,
    const float* __restrict__ weight_ptr,
    const float* __restrict__ bias_ptr,
    float* __restrict__ output_ptr,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Each warp (32 threads) computes one output element.
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / 32;  // one warp per output element
    int lane = threadIdx.x & 31;          // lane index within the warp

    int total_outputs = batch_size * out_channels * output_length;
    if (warp_id >= total_outputs) return;

    // Decompose the linear output index into (b, oc, o)
    int b = warp_id / (out_channels * output_length);
    int rem = warp_id % (out_channels * output_length);
    int oc = rem / output_length;
    int o = rem % output_length;

    float sum = 0.0f;
    int total_iter = in_channels * kernel_size; // Loop over all (in_channel, kernel pos) pairs

    // Each thread in the warp processes a subset of the iterations
    for (int idx = lane; idx < total_iter; idx += 32) {
        int ic = idx / kernel_size;
        int k = idx % kernel_size;

        int i_pos = o + padding - k * dilation;
        if ((i_pos % stride) != 0) continue;
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;

        int x_index = b * in_channels * input_length + ic * input_length + i;
        int w_index = ic * out_channels * kernel_size + oc * kernel_size + k;
        sum += x_ptr[x_index] * weight_ptr[w_index];
    }

    // Perform warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Lane 0 writes the final result
    if (lane == 0) {
        if (bias_ptr) {
            sum += bias_ptr[oc];
        }
        int out_index = b * out_channels * output_length + oc * output_length + o;
        output_ptr[out_index] = sum;
    }
}

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, in_channels, input_length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");

    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_contig;
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_contig = bias->contiguous();
        TORCH_CHECK(bias_contig.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_contig.dim() == 1, "bias must be 1D");
        bias_ptr = bias_contig.data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    // Compute output length
    int output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    // Each output element is computed by one warp (32 threads)
    int total_outputs = batch_size * out_channels * output_length;
    int total_threads = total_outputs * 32;  // one warp per output
    int threads_per_block = 256;  // e.g., 256 threads per block (8 warps per block)
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    conv_transpose1d_warp_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "ConvTranspose1D forward with warp-level reduction (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
