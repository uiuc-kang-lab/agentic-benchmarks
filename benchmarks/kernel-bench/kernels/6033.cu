#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_shared_shfl_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    extern __shared__ float s_in[];
    
    int tid = threadIdx.x;
    int channel = blockIdx.y;
    int batch = blockIdx.z;
    int o = blockIdx.x * blockDim.x + tid;
    
    if (o >= output_length || channel >= in_channels || batch >= batch_size) return;

    // Calculate input window for this block
    int block_start = blockIdx.x * blockDim.x * stride - padding;
    int block_end = (blockIdx.x * blockDim.x + blockDim.x) * stride + kernel_size - 1 - padding;
    int load_size = block_end - block_start;

    // Cooperative load into shared memory
    for (int i = tid; i < load_size; i += blockDim.x) {
        int pos = block_start + i;
        if (pos >= 0 && pos < input_length) {
            s_in[i] = input[batch * in_channels * input_length + channel * input_length + pos];
        } else {
            s_in[i] = 0.0f;
        }
    }
    __syncthreads();

    // Calculate position in shared memory
    int window_start = o * stride - padding - block_start;
    float sum = 0.0f;

    // Main summation with warp-level optimization
    for (int k = 0; k < kernel_size; ++k) {
        if (window_start + k >= 0 && window_start + k < load_size) {
            sum += s_in[window_start + k];
        }
    }

    // Warp-level reduction for final sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid % 32 == 0) {
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

    dim3 threads(256);
    dim3 grid(
        (output_length + threads.x - 1) / threads.x,
        in_channels,
        batch_size
    );

    int shared_mem = (threads.x * stride + kernel_size) * sizeof(float);
    
    avg_pool1d_shared_shfl_kernel<<<grid, threads, shared_mem>>>(
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
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward (CUDA)");
}
