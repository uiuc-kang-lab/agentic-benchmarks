#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

constexpr int BLOCK_SIZE = 256;
constexpr int NUM_STREAMS = 4;  // Number of concurrent streams

__global__ void avg_pool3d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding,
    int batch_offset) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int elements_per_batch = channels * out_d * out_h * out_w;
    int total_elements = elements_per_batch;
    
    while (index < total_elements) {
        int w_out = index % out_w;
        int tmp = index / out_w;
        int h_out = tmp % out_h;
        tmp = tmp / out_h;
        int d_out = tmp % out_d;
        tmp = tmp / out_d;
        int c = tmp % channels;
        int n = batch_offset;  // Use the batch offset provided

        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        int d_end = d_start + kernel_size;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;

        int d_start_clamped = max(0, d_start);
        int h_start_clamped = max(0, h_start);
        int w_start_clamped = max(0, w_start);
        int d_end_clamped = min(d_end, in_d);
        int h_end_clamped = min(h_end, in_h);
        int w_end_clamped = min(w_end, in_w);

        float sum = 0.0f;
        #pragma unroll
        for (int d = d_start_clamped; d < d_end_clamped; ++d) {
            #pragma unroll
            for (int h = h_start_clamped; h < h_end_clamped; ++h) {
                #pragma unroll
                for (int w = w_start_clamped; w < w_end_clamped; ++w) {
                    int input_index = (((n * channels + c) * in_d + d) * in_h + h) * in_w + w;
                    sum += input[input_index];
                }
            }
        }

        int pool_volume = kernel_size * kernel_size * kernel_size;
        int output_index = index + batch_offset * elements_per_batch;
        output[output_index] = sum / static_cast<float>(pool_volume);
        
        index += blockDim.x * gridDim.x;
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);
    
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int elements_per_batch = channels * out_d * out_h * out_w;
    int threads = BLOCK_SIZE;
    int blocks = (elements_per_batch + threads - 1) / threads;
    blocks = min(blocks, 65535);

    // Process batches using multiple streams
    for (int n = 0; n < batch_size; ++n) {
        int stream_idx = n % NUM_STREAMS;
        
        avg_pool3d_forward_kernel<<<blocks, threads, 0, streams[stream_idx]>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels,
            in_d, in_h, in_w,
            out_d, out_h, out_w,
            kernel_size, stride, padding,
            n);
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with streams");
}