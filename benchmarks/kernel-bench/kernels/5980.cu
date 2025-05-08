#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void coop_shared_avg_pool1d_kernel(
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
    
    int tid = threadIdx.x;
    int block_o_start = blockIdx.x * blockDim.x * 4;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    // Calculate input segment for this block
    int input_start = block_o_start * stride - padding;
    int input_end = (block_o_start + blockDim.x * 4 - 1) * stride + kernel_size - 1 - padding;
    input_start = max(input_start, 0);
    input_end = min(input_end, input_length - 1);
    int input_segment_length = input_end - input_start + 1;

    // Cooperative load into shared memory
    for (int i = tid; i < input_segment_length; i += blockDim.x) {
        int pos_input = input_start + i;
        shared_input[i] = (pos_input < input_length && pos_input >= 0)
            ? input[batch * in_channels * input_length + channel * input_length + pos_input]
            : 0.0f;
    }
    __syncthreads();

    // Process 4 outputs per thread
    int o_base = block_o_start + tid * 4;
    for (int i = 0; i < 4; ++i) {
        int o = o_base + i;
        if (o >= output_length) break;

        float sum = 0.0f;
        int window_start = o * stride - padding;
        for (int k = 0; k < kernel_size; ++k) {
            int pos_shared = window_start + k - input_start;
            if (pos_shared >= 0 && pos_shared < input_segment_length)
                sum += shared_input[pos_shared];
        }
        output[batch * in_channels * output_length + channel * output_length + o] = sum / kernel_size;
    }
}

torch::Tensor coop_shared_avg_pool1d_forward(
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
        (output_length + threads.x * 4 - 1) / (threads.x * 4),
        in_channels,
        batch_size
    );

    int outputs_per_block = threads.x * 4;
    int max_input_segment = (outputs_per_block - 1) * stride + kernel_size;
    size_t shared_size = max_input_segment * sizeof(float);

    coop_shared_avg_pool1d_kernel<<<grid, threads, shared_size>>>(
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
    m.def("forward", &coop_shared_avg_pool1d_forward, "Optimized 1D Average Pooling with Cooperative Shared Memory");
}
