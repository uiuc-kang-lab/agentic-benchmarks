#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define VEC_SIZE 4

typedef float4 vec_type;

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
    
    const int channel = blockIdx.y;
    const int batch = blockIdx.z;
    const int tid = threadIdx.x;

    /* Grid configuration ensures valid channel and batch indices. */

    const int block_start_out = blockIdx.x * BLOCK_SIZE;
    const int block_end_out = min(block_start_out + BLOCK_SIZE, output_length);
    
    const int global_in_start = block_start_out * stride - padding;
    const int global_in_end = (block_end_out - 1) * stride + kernel_size - padding;
    const int eff_in_start = max(global_in_start, 0);
    const int eff_in_end = min(global_in_end, input_length);
    const int in_tile_size = eff_in_end - eff_in_start;

    // Vectorized loading of input tile (4 elements per thread)
    const int vec_count = (in_tile_size + VEC_SIZE - 1) / VEC_SIZE;
    const vec_type *in_vec = reinterpret_cast<const vec_type*>(input + batch * in_channels * input_length + channel * input_length + eff_in_start);
    
    for (int i = tid; i < vec_count; i += blockDim.x) {
        const int base_idx = i * VEC_SIZE;
        vec_type loaded;
        if (base_idx + VEC_SIZE <= in_tile_size) {
            loaded = in_vec[i];
        } else {
            #pragma unroll
            for (int v=0; v<VEC_SIZE; ++v) {
                const int idx = min(base_idx + v, in_tile_size-1);
                ((float*)&loaded)[v] = (eff_in_start + idx < input_length) ? input[batch * in_channels * input_length + channel * input_length + eff_in_start + idx] : 0.0f;
            }
        }
        #pragma unroll
        for (int v=0; v<VEC_SIZE; ++v) {
            const int mem_idx = i * VEC_SIZE + v;
            if (mem_idx < in_tile_size) {
                shmem[mem_idx] = ((float*)&loaded)[v];
            }
        }
    }
    __syncthreads();

    // Process outputs
    for (int o = block_start_out + tid; o < block_end_out; o += blockDim.x) {
        float sum = 0.0f;
        const int shmem_offset = o * stride - padding - eff_in_start;
        
        for (int k=0; k<kernel_size; ++k) {
            const int rel_pos = shmem_offset + k;
            if (rel_pos >= 0 && rel_pos < in_tile_size) {
                sum += shmem[rel_pos];
            }
        }
        
        const int out_idx = batch * in_channels * output_length + channel * output_length + o;
        output[out_idx] = sum / kernel_size;
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

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    dim3 threads(BLOCK_SIZE);
    dim3 grid(
        (output_length + BLOCK_SIZE - 1) / BLOCK_SIZE,
        in_channels,
        batch_size
    );

    const int shared_mem_size = (BLOCK_SIZE * stride + kernel_size + VEC_SIZE) * sizeof(float);

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
    m.def("forward", &avg_pool1d_forward, "1D AvgPool with vectorized memory loads (CUDA)");
}
