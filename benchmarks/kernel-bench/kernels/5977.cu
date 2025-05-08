#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void shared_mem_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int input_length,
    const int output_length,
    const int batch_size,
    const int in_channels) {

    extern __shared__ float shared_input[];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * in_channels * output_length;

    if (idx >= total_elements) return;

    const int o = idx % output_length;
    const int channel = (idx / output_length) % in_channels;
    const int batch = idx / (output_length * in_channels);

    const int input_batch_offset = batch * in_channels * input_length;
    const int input_channel_offset = channel * input_length;
    const int input_base = input_batch_offset + input_channel_offset;

    const int thread_input_offset = threadIdx.x * (kernel_size + stride);
    const int start_idx = o * stride - padding;

    float sum = 0.0f;

    if (threadIdx.x < kernel_size + stride) {
        shared_input[thread_input_offset] = (start_idx + threadIdx.x < input_length && start_idx + threadIdx.x >= 0) 
            ? input[input_base + start_idx + threadIdx.x] 
            : 0.0f;
    }

    __syncthreads();

    const int shared_start = max(0, -start_idx);
    const int shared_end = min(kernel_size, input_length - start_idx);

    #pragma unroll
    for (int k = shared_start; k < shared_end; ++k) {
        sum += shared_input[thread_input_offset + k];
    }

    output[idx] = sum / kernel_size;
}

torch::Tensor shared_mem_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    const int total_elements = batch_size * in_channels * output_length;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    const int shared_mem_size = threads * (kernel_size + stride) * sizeof(float);

    shared_mem_avg_pool1d_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &shared_mem_avg_pool1d_forward, "Shared Memory 1D Average Pooling forward (CUDA)");
}