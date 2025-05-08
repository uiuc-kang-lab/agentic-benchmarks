#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    const int o = threadIdx.x + blockIdx.x * blockDim.x;
    const int channel = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch = threadIdx.z + blockIdx.z * blockDim.z;

    // Early exit for threads outside bounds
    if (o >= output_length || channel >= in_channels || batch >= batch_size) return;

    // Pre-compute input base offset for this thread
    const int input_batch_offset = batch * in_channels * input_length;
    const int input_channel_offset = channel * input_length;
    const int base_idx = input_batch_offset + input_channel_offset;

    // Pre-compute the start position of the pooling window
    const int window_start = o * stride - padding;
    
    // Pre-compute window boundaries
    const int valid_start = max(0, window_start);
    const int valid_end = min(input_length, window_start + kernel_size);
    
    // Count valid elements for accurate averaging
    const int valid_elements = valid_end - valid_start;
    
    float sum = 0.0f;
    
    // Main computation loop - now divergence-free within each warp
    // since boundary conditions are pre-computed
    #pragma unroll 4
    for (int k = 0; k < kernel_size; ++k) {
        const int pos_input = window_start + k;
        // Use predication instead of branching
        const bool valid = (pos_input >= 0 && pos_input < input_length);
        const float val = valid ? input[base_idx + pos_input] : 0.0f;
        sum += val;
    }

    // Compute output index
    const int output_idx = batch * in_channels * output_length + 
                          channel * output_length + o;
    
    // Write result - all threads in a warp will execute this
    output[output_idx] = sum / kernel_size;
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

    // Optimize thread block dimensions for better occupancy
    // and maintain good memory access patterns
    dim3 threads(32, 8, 4);  // 32 threads per warp, utilizing 3D blocking
    dim3 grid(
        (output_length + threads.x - 1) / threads.x,
        (in_channels + threads.y - 1) / threads.y,
        (batch_size + threads.z - 1) / threads.z
    );

    avg_pool1d_kernel<<<grid, threads>>>(
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