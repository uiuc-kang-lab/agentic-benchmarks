#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constant memory for frequently accessed pooling parameters
// pool_params[0] = kernel_size, pool_params[1] = stride, pool_params[2] = padding,
// pool_params[3] = input_length, pool_params[4] = output_length
__constant__ int pool_params[5];

// Define shorthand macros for accessing constant memory parameters
#define KERNEL_SIZE (pool_params[0])
#define STRIDE      (pool_params[1])
#define PADDING     (pool_params[2])
#define INPUT_LENGTH (pool_params[3])
#define OUTPUT_LENGTH (pool_params[4])

__global__ void avg_pool1d_kernel_const(
    const float *input,
    float *output,
    int batch_size,
    int in_channels) {

    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    if (o >= OUTPUT_LENGTH || channel >= in_channels || batch >= batch_size)
        return;

    float sum = 0.0f;
    for (int k = 0; k < KERNEL_SIZE; ++k) {
        int pos_padded = o * STRIDE + k;
        int pos_input = pos_padded - PADDING;
        
        if (pos_input >= 0 && pos_input < INPUT_LENGTH) {
            int input_idx = batch * in_channels * INPUT_LENGTH + channel * INPUT_LENGTH + pos_input;
            sum += input[input_idx];
        }
    }

    int output_idx = batch * in_channels * OUTPUT_LENGTH + channel * OUTPUT_LENGTH + o;
    output[output_idx] = sum / KERNEL_SIZE;
}

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

    // Copy pooling parameters to constant memory for faster access on GPU cores
    int h_pool_params[5] = { kernel_size, stride, padding, input_length, output_length };
    cudaMemcpyToSymbol(pool_params, h_pool_params, 5 * sizeof(int), 0, cudaMemcpyHostToDevice);

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    dim3 threads(256);
    dim3 grid(
        (output_length + threads.x - 1) / threads.x,
        in_channels,
        batch_size
    );

    avg_pool1d_kernel_const<<<grid, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward with constant memory (CUDA)");
}
