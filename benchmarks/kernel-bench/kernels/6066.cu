#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__device__ __forceinline__ float compute_sum(const float* __restrict__ shmem, int shmem_offset, int kernel_size, int in_tile_size) {
    float sum = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < kernel_size; ++k) {
        int rel_pos = shmem_offset + k;
        if (rel_pos >= 0 && rel_pos < in_tile_size) {
            sum = __fmaf_rn(1.0f, shmem[rel_pos], sum);  // Using fast math intrinsic
        }
    }
    return sum;
}

__device__ void load_to_shared_memory(const float* __restrict__ input, float* shmem, int global_in_start, int in_tile_size, int input_length, int batch, int channel, int in_channels, int tid) {
    for (int i = tid; i < in_tile_size; i += blockDim.x) {
        int global_idx = global_in_start + i;
        float val = (global_idx < input_length) ? input[batch * in_channels * input_length + channel * input_length + global_idx] : 0.0f;
        shmem[i] = val;
    }
    __syncthreads();
}

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* output,
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

    // Load input tile into shared memory
    load_to_shared_memory(input, shmem, global_in_start, in_tile_size, input_length, batch, channel, in_channels, tid);

    // Process outputs covered by this block
    for (int o = block_start_out + tid; o < block_end_out; o += blockDim.x) {
        int shmem_offset = o * stride - padding - global_in_start;
        float sum = compute_sum(shmem, shmem_offset, kernel_size, in_tile_size);
        output[batch * in_channels * output_length + channel * output_length + o] = sum / kernel_size;
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

    dim3 threads(BLOCK_SIZE);
    dim3 grid(
        (output_length + BLOCK_SIZE - 1) / BLOCK_SIZE,
        in_channels,
        batch_size
    );

    int shared_mem_size = (BLOCK_SIZE * stride + kernel_size -1 + BLOCK_SIZE) * sizeof(float);

    avg_pool1d_kernel<<<grid, threads, shared_mem_size>>>(
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
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling with modular device functions (CUDA)");
}
