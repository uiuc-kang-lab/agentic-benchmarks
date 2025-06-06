#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define a structure to hold constant parameters
struct ConstParams {
    int kernel_size;
    int stride;
    int padding;
    float inv_kernel; // 1.0f / kernel_size
};

// Declare constant memory for pooling parameters
__constant__ ConstParams const_params;

// CUDA kernel for 1D Average Pooling using constant memory for frequently accessed parameters
__global__ void const_memory_avg_pool1d_kernel(
    const float *input,
    float *output,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    if (o >= output_length || channel >= in_channels || batch >= batch_size) return;

    float sum = 0.0f;
    // Use the kernel parameters from constant memory
    for (int k = 0; k < const_params.kernel_size; ++k) {
        int pos_padded = o * const_params.stride + k;
        int pos_input = pos_padded - const_params.padding;
        if (pos_input >= 0 && pos_input < input_length) {
            int input_idx = batch * in_channels * input_length 
                            + channel * input_length 
                            + pos_input;
            sum += input[input_idx];
        }
    }

    int output_idx = batch * in_channels * output_length 
                     + channel * output_length 
                     + o;
    output[output_idx] = sum * const_params.inv_kernel;
}

// Host function to launch the CUDA kernel
torch::Tensor const_memory_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Prepare constant parameters and copy to device constant memory
    ConstParams h_params;
    h_params.kernel_size = kernel_size;
    h_params.stride = stride;
    h_params.padding = padding;
    h_params.inv_kernel = 1.0f / kernel_size;
    cudaMemcpyToSymbol(const_params, &h_params, sizeof(ConstParams));

    // Choose block and grid dimensions
    dim3 threads(256);
    dim3 grid(
        (output_length + threads.x - 1) / threads.x,
        in_channels,
        batch_size
    );

    const_memory_avg_pool1d_kernel<<<grid, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        input_length,
        output_length,
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &const_memory_avg_pool1d_forward, "1D Average Pooling forward (CUDA) with constant memory");
}
