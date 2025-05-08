#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Store frequently accessed parameters in constant memory
__constant__ int cKernelSize;
__constant__ int cStride;
__constant__ int cPadding;
__constant__ int cInputLength;
__constant__ int cOutputLength;

// CUDA Kernel using constant memory for pooling parameters
__global__ void avg_pool1d_kernel(const float *input, float *output, int batch_size, int in_channels) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    if (o >= cOutputLength || channel >= in_channels || batch >= batch_size)
        return;

    float sum = 0.0f;
    for (int k = 0; k < cKernelSize; ++k) {
        int pos_padded = o * cStride + k;
        int pos_input = pos_padded - cPadding;
        if (pos_input >= 0 && pos_input < cInputLength) {
            int input_idx = batch * in_channels * cInputLength + channel * cInputLength + pos_input;
            sum += input[input_idx];
        }
    }

    int output_idx = batch * in_channels * cOutputLength + channel * cOutputLength + o;
    output[output_idx] = sum / cKernelSize;
}

// Host function to launch the CUDA kernel
torch::Tensor avg_pool1d_forward(
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

    // Copy pooling parameters to constant memory
    cudaMemcpyToSymbol(cKernelSize, &kernel_size, sizeof(int));
    cudaMemcpyToSymbol(cStride, &stride, sizeof(int));
    cudaMemcpyToSymbol(cPadding, &padding, sizeof(int));
    cudaMemcpyToSymbol(cInputLength, &input_length, sizeof(int));
    cudaMemcpyToSymbol(cOutputLength, &output_length, sizeof(int));

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    dim3 threads(256);
    dim3 grid(
        (output_length + threads.x - 1) / threads.x,
        in_channels,
        batch_size
    );

    avg_pool1d_kernel<<<grid, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward (CUDA) using constant memory");
}
