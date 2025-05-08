#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Maximum kernel size supported for constant memory optimization
#define MAX_KERNEL_SIZE 64

// Constant memory array to store pooling offsets
__constant__ int pool_offsets[MAX_KERNEL_SIZE];

// CUDA kernel for 1D Average Pooling using constant memory for pooling offsets
__global__ void avg_pool1d_const_kernel(
    const float *input,
    float *output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int input_length,
    const int output_length,
    const int batch_size,
    const int in_channels) {

    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    if (o >= output_length || channel >= in_channels || batch >= batch_size)
        return;

    float sum = 0.0f;
    int base_idx = batch * in_channels * input_length + channel * input_length;
    int start_index = o * stride;

    // Loop over the kernel window using offsets stored in constant memory
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int pos_padded = start_index + pool_offsets[k];
        int pos_input = pos_padded - padding;
        if (pos_input >= 0 && pos_input < input_length) {
            sum += input[base_idx + pos_input];
        }
    }

    int output_idx = batch * in_channels * output_length + channel * output_length + o;
    output[output_idx] = sum / kernel_size;
}

// Host function to launch the CUDA kernel
// This function copies the pooling offsets into constant memory before launching the kernel

torch::Tensor avg_pool1d_const_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid pooling parameters");
    TORCH_CHECK(kernel_size <= MAX_KERNEL_SIZE, "Kernel size exceeds constant memory limits");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Prepare pooling offsets (0, 1, 2, ..., kernel_size-1) and copy to constant memory
    int h_pool_offsets[MAX_KERNEL_SIZE];
    for (int i = 0; i < kernel_size; ++i) {
        h_pool_offsets[i] = i;
    }
    cudaMemcpyToSymbol(pool_offsets, h_pool_offsets, kernel_size * sizeof(int), 0, cudaMemcpyHostToDevice);

    dim3 threads(256);
    dim3 grid((output_length + threads.x - 1) / threads.x, in_channels, batch_size);

    avg_pool1d_const_kernel<<<grid, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        batch_size,
        in_channels);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_const_forward, "1D Average Pooling forward (CUDA) with constant memory");
}
