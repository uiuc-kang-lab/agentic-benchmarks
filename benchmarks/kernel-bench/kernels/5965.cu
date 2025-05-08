#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void shared_unroll_avg_pool1d_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    extern __shared__ float s_data[];

    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    if (o >= output_length || channel >= in_channels || batch >= batch_size) return;

    int o_start = blockIdx.x * blockDim.x;
    int o_end = min(o_start + blockDim.x, output_length);
    int i_start = o_start * stride - padding;
    int i_end = (o_end - 1) * stride + kernel_size - padding;
    int smem_size = i_end - i_start;

    for (int i = threadIdx.x; i < smem_size; i += blockDim.x) {
        int pos_input = i_start + i;
        s_data[i] = (pos_input >= 0 && pos_input < input_length) ? 
            input[batch * in_channels * input_length + channel * input_length + pos_input] : 0.0f;
    }
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int smem_idx = o * stride + k - padding - i_start;
        if (smem_idx >= 0 && smem_idx < smem_size) {
            sum += s_data[smem_idx];
        }
    }

    output[batch * in_channels * output_length + channel * output_length + o] = sum / kernel_size;
}

torch::Tensor shared_unroll_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    dim3 threads(256);
    dim3 grid(
        (output_length + threads.x - 1) / threads.x,
        in_channels,
        batch_size
    );

    int smem_size = (threads.x * stride + kernel_size - 1) * sizeof(float);
    shared_unroll_avg_pool1d_kernel<<<grid, threads, smem_size>>>(
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
    m.def("forward", &shared_unroll_avg_pool1d_forward, "1D Average Pooling with shared memory and loop unrolling");
}