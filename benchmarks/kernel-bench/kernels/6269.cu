#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool3d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    // Shared memory for partial sums within the block
    __shared__ float shared_data[32 * 8];
    
    // Decode the combined (n, c, d_out) from blockIdx.z
    int idx = blockIdx.z;
    int d_out = idx % out_d;
    idx /= out_d;
    int c = idx % channels;
    int n = idx / channels;

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    
    // Output coordinates
    int h_out = blockIdx.y * blockDim.y + ty;
    int w_out = blockIdx.x * blockDim.x + tx;
    
    if (h_out >= out_h || w_out >= out_w) return;

    // Calculate input window boundaries
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    
    // Clamp boundaries
    int d_start_clamped = max(d_start, 0);
    int h_start_clamped = max(h_start, 0);
    int w_start_clamped = max(w_start, 0);
    int d_end_clamped = min(d_start + kernel_size, in_d);
    int h_end_clamped = min(h_start + kernel_size, in_h);
    int w_end_clamped = min(w_start + kernel_size, in_w);

    // Initialize partial sum
    float partial_sum = 0.0f;
    
    // Base offset for current (n,c) slice
    int baseOffset = (n * channels + c) * in_d;

    // Accumulate values for this thread's output element
    #pragma unroll
    for (int d = d_start_clamped; d < d_end_clamped; d++) {
        int d_offset = (baseOffset + d) * in_h * in_w;
        #pragma unroll
        for (int h = h_start_clamped; h < h_end_clamped; h++) {
            int row_start = d_offset + h * in_w + w_start_clamped;
            #pragma unroll
            for (int w = 0; w < w_end_clamped - w_start_clamped; w++) {
                partial_sum += input[row_start + w];
            }
        }
    }

    // Compute and store final averaged result directly
    int pool_volume = kernel_size * kernel_size * kernel_size;
    int output_idx = (((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out;
    output[output_idx] = partial_sum / static_cast<float>(pool_volume);

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

    // Configure thread block and grid dimensions
    dim3 block(32, 8, 1);
    dim3 grid((out_w + block.x - 1) / block.x,
              (out_h + block.y - 1) / block.y,
              batch_size * channels * out_d);

    avg_pool3d_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with shared memory");
}