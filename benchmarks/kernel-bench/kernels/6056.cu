#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses a linear mapping of output elements so that consecutive threads
// write to consecutive (contiguous) memory positions. This ensures memory coalescing
// for global memory accesses when writing the output. The input is accessed in a similar
// fashion by computing the corresponding batch, channel, and spatial index.

__global__ void avg_pool1d_vectorized_kernel(
    const float * __restrict__ input,
    float * __restrict__ output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int channels,
    int batch_size) {

    // Compute a linear index for the output element
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * output_length;
    if (index >= total_outputs) return;

    // Map the linear index to (b, c, o) coordinates
    int o = index % output_length;
    int temp = index / output_length;
    int c = temp % channels;
    int b = temp / channels;

    // Calculate the base pointer for this (b, c) slice in the input
    // The input tensor is stored in [batch, channels, input_length] (contiguous in input_length)
    int input_offset = b * channels * input_length + c * input_length;

    float sum = 0.f;
    int start = o * stride - padding;
    // Loop over the pooling window; assume kernel_size is small so unrolling may help
    #pragma unroll
    for (int k = 0; k < kernel_size; k++) {
        int pos = start + k;
        if (pos >= 0 && pos < input_length) {
            sum += input[input_offset + pos];
        }
    }
    
    // Write the result to the output with coalesced global memory access
    output[index] = sum / kernel_size;
}

torch::Tensor avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {
    
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, channels, length)");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_length}, x.options());

    int total_outputs = batch_size * channels * output_length;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;

    avg_pool1d_vectorized_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        channels,
        batch_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward with vectorized memory accesses (CUDA)");
}
