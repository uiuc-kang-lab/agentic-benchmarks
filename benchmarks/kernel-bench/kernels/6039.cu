#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Declare constant memory for frequently accessed parameters
__constant__ int d_kernel_size;
__constant__ int d_stride;
__constant__ int d_padding;
__constant__ int d_input_length;
__constant__ int d_output_length;
__constant__ int d_in_channels;

// Kernel using constant memory for pooling parameters
__global__ void avg_pool1d_kernel_const(
    const float *input,
    float *output,
    int batch_size) {  // batch_size is passed as a kernel argument

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    if (channel >= d_in_channels || batch >= batch_size) return;

    // Each thread processes multiple output indices using a grid-stride loop
    for (int o = idx; o < d_output_length; o += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < d_kernel_size; ++k) {
            int pos_padded = o * d_stride + k;
            int pos_input = pos_padded - d_padding;
            if (pos_input >= 0 && pos_input < d_input_length) {
                int input_idx = batch * d_in_channels * d_input_length + channel * d_input_length + pos_input;
                sum += input[input_idx];
            }
        }
        int output_idx = batch * d_in_channels * d_output_length + channel * d_output_length + o;
        output[output_idx] = sum / d_kernel_size;
    }
}

// Forward function copying parameters to constant memory and launching the kernel
torch::Tensor avg_pool1d_forward_const(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, channels, length)");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    // Copy pooling parameters to constant memory
    cudaMemcpyToSymbol(d_kernel_size, &kernel_size, sizeof(int));
    cudaMemcpyToSymbol(d_stride, &stride, sizeof(int));
    cudaMemcpyToSymbol(d_padding, &padding, sizeof(int));
    cudaMemcpyToSymbol(d_input_length, &input_length, sizeof(int));
    cudaMemcpyToSymbol(d_output_length, &output_length, sizeof(int));
    cudaMemcpyToSymbol(d_in_channels, &in_channels, sizeof(int));

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Configure threads and grid dimensions
    dim3 threads(256);
    dim3 grid(
        (output_length + threads.x - 1) / threads.x,
        in_channels,
        batch_size
    );

    avg_pool1d_kernel_const<<<grid, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward_const, "1D Average Pooling forward with constant memory (CUDA)");
}
