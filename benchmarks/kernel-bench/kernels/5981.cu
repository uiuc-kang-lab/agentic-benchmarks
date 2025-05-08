#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void hybrid_avg_pool1d_kernel(
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
    
    const int ELEMENTS_PER_THREAD = 4;
    int o_start = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x * ELEMENTS_PER_THREAD;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    if (channel >= in_channels || batch >= batch_size) return;

    int input_offset = batch * in_channels * input_length + channel * input_length;
    int output_offset = batch * in_channels * output_length + channel * output_length;

    int shared_idx = threadIdx.x;
    int max_shared_elements = blockDim.x * (kernel_size + (ELEMENTS_PER_THREAD-1) * stride);
    
    while (shared_idx < max_shared_elements) {
        int pos_padded = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD * stride + shared_idx;
        int pos_input = pos_padded - padding;
        
        shared_input[shared_idx] = (pos_input >= 0 && pos_input < input_length) 
            ? input[input_offset + pos_input] 
            : 0.0f;
            
        shared_idx += blockDim.x;
    }
    
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int o = o_start + i;
        if (o >= output_length) break;

        float sum = 0.0f;
        int base_idx = threadIdx.x * stride + i * stride;
        
        #pragma unroll
        for (int k = 0; k < kernel_size; k += 4) {
            float4 values = *reinterpret_cast<const float4*>(&shared_input[base_idx + k]);
            sum += values.x + values.y + values.z + values.w;
        }
        
        for (int k = (kernel_size/4)*4; k < kernel_size; ++k) {
            sum += shared_input[base_idx + k];
        }

        output[output_offset + o] = sum / kernel_size;
    }
}

torch::Tensor hybrid_avg_pool1d_forward(
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

    const int THREAD_BLOCK_SIZE = 128;
    const int ELEMENTS_PER_THREAD = 4;
    
    dim3 threads(THREAD_BLOCK_SIZE);
    dim3 grid(
        (output_length + threads.x * ELEMENTS_PER_THREAD - 1) / (threads.x * ELEMENTS_PER_THREAD),
        in_channels,
        batch_size
    );

    size_t shared_memory_size = (THREAD_BLOCK_SIZE * (kernel_size + (ELEMENTS_PER_THREAD-1) * stride)) * sizeof(float);

    hybrid_avg_pool1d_kernel<<<grid, threads, shared_memory_size>>>(
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