#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128
#define NUM_STREAMS 4

__global__ void avg_pool1d_kernel(
    const float *__restrict__ input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    extern __shared__ float shmem[];

    int channel = blockIdx.y;
    int batch = blockIdx.z;
    int tid = threadIdx.x;

    if (channel >= in_channels || batch >= batch_size) return;

    int block_start_out = blockIdx.x * BLOCK_SIZE;
    int block_end_out = min(block_start_out + BLOCK_SIZE, output_length);

    int global_in_start = block_start_out * stride - padding;
    int global_in_end = (block_end_out - 1) * stride + kernel_size - padding;
    global_in_start = max(global_in_start, 0);
    global_in_end = min(global_in_end, input_length);
    
    int in_tile_size = global_in_end - global_in_start;

    int elements_per_thread = (in_tile_size + blockDim.x - 1) / blockDim.x;
    for(int i = 0; i < elements_per_thread; ++i) {
        int idx = tid + i * blockDim.x;
        if(idx < in_tile_size) {
            int global_idx = global_in_start + idx;
            shmem[idx] = (global_idx < input_length)
                ? input[batch * in_channels * input_length + channel * input_length + global_idx]
                : 0.0f;
        }
    }
    __syncthreads();

    for(int o = block_start_out + tid; o < block_end_out; o += blockDim.x) {
        float sum = 0.0f;
        int shmem_offset = o * stride - padding - global_in_start;
        
        #pragma unroll 4
        for(int k = 0; k < kernel_size; ++k) {
            int rel_pos = shmem_offset + k;
            if(rel_pos >= 0 && rel_pos < in_tile_size) {
                sum += shmem[rel_pos];
            }
        }
        output[batch * in_channels * output_length + channel * output_length + o] = (kernel_size > 0) ? sum / kernel_size : 0.0f;
    }
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

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for(int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int batches_per_stream = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS;
    for(int s = 0; s < NUM_STREAMS; ++s) {
        int stream_batches_start = s * batches_per_stream;
        int stream_batches_end = std::min((s+1) * batches_per_stream, batch_size);
        int current_batches = stream_batches_end - stream_batches_start;

        if(current_batches <= 0) continue;

        dim3 threads(BLOCK_SIZE);
        dim3 grid(
            (output_length + BLOCK_SIZE - 1) / BLOCK_SIZE,
            in_channels,
            current_batches
        );

        int max_shared_mem = ((BLOCK_SIZE-1)*stride + kernel_size) * sizeof(float);

        avg_pool1d_kernel<<<grid, threads, max_shared_mem, streams[s]>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            kernel_size,
            stride,
            padding,
            input_length,
            output_length,
            current_batches,
            in_channels
        );
    }

    for(int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling with stream optimization (CUDA)");
}
