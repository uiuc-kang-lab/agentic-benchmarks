#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4

__global__ void avg_pool1d_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels,
    int batch_offset,
    int batch_chunk_size) {

    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    int batch = blockIdx.z + batch_offset;

    if (o >= output_length || channel >= in_channels || batch >= (batch_offset + batch_chunk_size)) return;

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int pos_padded = o * stride + k;
        int pos_input = pos_padded - padding;
        
        if (pos_input >= 0 && pos_input < input_length) {
            int input_idx = batch * in_channels * input_length + channel * input_length + pos_input;
            sum += __ldg(&input[input_idx]);
        }
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

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threads(256);
    int batch_chunk_size = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS;

    for (int i = 0; i < NUM_STREAMS; i++) {
        int batch_offset = i * batch_chunk_size;
        int current_chunk_size = min(batch_chunk_size, batch_size - batch_offset);
        
        if (current_chunk_size <= 0) break;

        dim3 grid(
            (output_length + threads.x - 1) / threads.x,
            in_channels,
            current_chunk_size
        );

        avg_pool1d_kernel<<<grid, threads, 0, streams[i]>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            kernel_size,
            stride,
            padding,
            input_length,
            output_length,
            batch_size,
            in_channels,
            batch_offset,
            current_chunk_size
        );
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward (CUDA)");
}