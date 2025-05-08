#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    extern __shared__ float shared_input[];
    
    const int tid = threadIdx.x;
    const int channel = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch = threadIdx.z + blockIdx.z * blockDim.z;
    const int block_start = blockIdx.x * blockDim.x;
    const int o = block_start + tid;

    // Calculate the input range needed for this block
    const int input_start = block_start * stride - padding;
    const int input_end = (block_start + blockDim.x - 1) * stride + kernel_size - padding;
    const int shared_mem_size = input_end - input_start;

    // Load input data into shared memory
    if (channel < in_channels && batch < batch_size) {
        const int base_idx = batch * in_channels * input_length + channel * input_length;
        
        for (int i = tid; i < shared_mem_size; i += blockDim.x) {
            int input_pos = input_start + i;
            if (input_pos >= 0 && input_pos < input_length) {
                shared_input[i] = input[base_idx + input_pos];
            } else {
                shared_input[i] = 0.0f;
            }
        }
    }
    
    __syncthreads();

    if (o >= output_length || channel >= in_channels || batch >= batch_size) return;

    float sum = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < kernel_size; ++k) {
        int shared_pos = (o - block_start) * stride + k;
        sum += shared_input[shared_pos];
    }

    output[batch * in_channels * output_length + channel * output_length + o] = sum / kernel_size;
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

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    dim3 threads(32, 8, 4);
    dim3 grid(
        (output_length + threads.x - 1) / threads.x,
        (in_channels + threads.y - 1) / threads.y,
        (batch_size + threads.z - 1) / threads.z
    );

    avg_pool1d_kernel<<<grid, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward (CUDA)");
}