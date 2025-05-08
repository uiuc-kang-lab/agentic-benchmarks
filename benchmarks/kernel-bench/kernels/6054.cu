#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

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
    
    // Determine input range needed for this block's outputs
    int global_in_start = block_start_out * stride - padding;
    int global_in_end = (block_end_out - 1) * stride + kernel_size - padding;
    global_in_start = max(global_in_start, 0);
    global_in_end = min(global_in_end, input_length);
    
    int in_tile_size = global_in_end - global_in_start;

    // Load input tile into shared memory with coalesced access
    for (int i = tid; i < in_tile_size; i += blockDim.x) {
        int global_idx = global_in_start + i;
        float val = (global_idx < input_length) ? input[batch * in_channels * input_length + channel * input_length + global_idx] : 0.0f;
        shmem[i] = val;
    }
    __syncthreads();

    // Process outputs covered by this block
    for (int o = block_start_out + tid; o < block_end_out; o += blockDim.x) {
        float sum = 0.0f;
        int shmem_offset = o * stride - padding - global_in_start;
        
        for (int k = 0; k < kernel_size; ++k) {
            int rel_pos = shmem_offset + k;
            if (rel_pos >= 0 && rel_pos < in_tile_size) {
                sum += shmem[rel_pos];
            }
        }

        output[batch * in_channels * output_length + channel * output_length + o] = sum / kernel_size;
    }
}

void avg_pool1d_forward(
    const torch::Tensor &x,
    torch::Tensor &output,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream) {
    
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    dim3 threads(BLOCK_SIZE);
    dim3 grid(
        (output_length + BLOCK_SIZE - 1) / BLOCK_SIZE,
        in_channels,
        batch_size
    );

    int shared_mem_size = (BLOCK_SIZE * stride + kernel_size -1 + BLOCK_SIZE) * sizeof(float);

    avg_pool1d_kernel<<<grid, threads, shared_mem_size, stream>>>(
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
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", [](const torch::Tensor &x, int kernel_size, int stride, int padding) {
        auto output = torch::empty({x.size(0), x.size(1), (x.size(2) + 2 * padding - kernel_size) / stride + 1}, x.options());
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        avg_pool1d_forward(x, output, kernel_size, stride, padding, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        return output;
    }, "1D Average Pooling with shared memory and streams (CUDA)");
}