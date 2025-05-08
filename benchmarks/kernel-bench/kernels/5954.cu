#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void shared_memory_avg_pool1d_kernel(
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

    int o_start = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    int input_offset = batch * in_channels * input_length + channel * input_length;
    int output_offset = batch * in_channels * output_length + channel * output_length;

    for (int i = 0; i < 4; i++) {
        int o = o_start + i;
        if (o >= output_length) break;

        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            int pos_padded = o * stride + k;
            int pos_input = pos_padded - padding;

            if (pos_input >= 0 && pos_input < input_length) {
                int input_idx = input_offset + pos_input;
                shared_input[threadIdx.x * kernel_size + k] = input[input_idx];
            } else {
                shared_input[threadIdx.x * kernel_size + k] = 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < kernel_size; ++k) {
            sum += shared_input[threadIdx.x * kernel_size + k];
        }

        output[output_offset + o] = sum / kernel_size;
    }
}

torch::Tensor shared_memory_avg_pool1d_forward(
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

    dim3 threads(64);  // Each thread handles 4 outputs
    dim3 grid(
        (output_length + threads.x * 4 - 1) / (threads.x * 4),
        in_channels,
        batch_size
    );

    size_t shared_memory_size = threads.x * kernel_size * sizeof(float);

    shared_memory_avg_pool1d_kernel<<<grid, threads, shared_memory_size>>>(
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
    m.def("forward", &shared_memory_avg_pool1d_forward, "1D Average Pooling forward (CUDA) with shared memory");
}