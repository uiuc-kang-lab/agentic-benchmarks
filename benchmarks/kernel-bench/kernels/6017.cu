#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Store pooling parameters in constant memory for quick access
// pool_params[0] = kernel_size, pool_params[1] = stride, pool_params[2] = padding,
// pool_params[3] = input_length, pool_params[4] = output_length
__constant__ int pool_params[5];

// Define macros for ease of use
#define KERNEL_SIZE   (pool_params[0])
#define STRIDE        (pool_params[1])
#define PADDING       (pool_params[2])
#define INPUT_LENGTH  (pool_params[3])
#define OUTPUT_LENGTH (pool_params[4])

// Efficient kernel using a grid-stride loop for flexible parallelization
__global__ void avg_pool1d_kernel(
    const float *input,
    float *output,
    int batch_size,
    int in_channels) {

    // Total number of output elements
    int totalElements = batch_size * in_channels * OUTPUT_LENGTH;

    // Use grid-stride loop to cover all elements
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalElements; index += blockDim.x * gridDim.x) {
        // Decompose linear index into batch, channel, and output index
        int tmp = index;
        int o = tmp % OUTPUT_LENGTH;
        tmp /= OUTPUT_LENGTH;
        int channel = tmp % in_channels;
        int batch = tmp / in_channels;

        // Compute starting index for pooling window considering padding
        int start = o * STRIDE - PADDING;
        float sum = 0.0f;
        
        // Sum over the pooling window
        for (int k = 0; k < KERNEL_SIZE; k++) {
            int pos = start + k;
            if (pos >= 0 && pos < INPUT_LENGTH) {
                int input_idx = batch * in_channels * INPUT_LENGTH + channel * INPUT_LENGTH + pos;
                sum += input[input_idx];
            }
        }

        // Write the averaged result
        output[index] = sum / KERNEL_SIZE;
    }
}

// Host function for average pooling forward pass
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

    // Copy parameters to constant memory for fast access
    int h_pool_params[5] = { kernel_size, stride, padding, input_length, output_length };
    cudaMemcpyToSymbol(pool_params, h_pool_params, 5 * sizeof(int));

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    int totalElements = batch_size * in_channels * output_length;
    int threads = 256;
    int blocks = (totalElements + threads - 1) / threads;

    avg_pool1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward with constant memory and grid-stride loop (CUDA)");
}
