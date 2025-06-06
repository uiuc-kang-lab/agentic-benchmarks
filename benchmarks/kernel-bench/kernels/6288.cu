#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

__global__ void avg_pool3d_forward_kernel_coalesced(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size, const int channels,
    const int in_d, const int in_h, const int in_w,
    const int out_d, const int out_h, const int out_w,
    const int kernel_size, const int stride, const int padding) {

    // Calculate total output size
    const int out_chw = out_d * out_h * out_w;
    const int total_elements = batch_size * channels * out_chw;
    
    // Process multiple elements per thread using grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements; 
         idx += blockDim.x * gridDim.x) {
        
        // Decompose index for coalesced memory access
        const int w_out = idx % out_w;
        const int h_out = (idx / out_w) % out_h;
        const int d_out = (idx / (out_w * out_h)) % out_d;
        const int c = (idx / out_chw) % channels;
        const int n = idx / (channels * out_chw);

        // Calculate input window boundaries
        const int d_start = max(d_out * stride - padding, 0);
        const int h_start = max(h_out * stride - padding, 0);
        const int w_start = max(w_out * stride - padding, 0);
        const int d_end = min(d_out * stride - padding + kernel_size, in_d);
        const int h_end = min(h_out * stride - padding + kernel_size, in_h);
        const int w_end = min(w_out * stride - padding + kernel_size, in_w);

        // Pre-compute base offset for input
        const int base_offset = ((n * channels + c) * in_d) * in_h * in_w;
        
        float sum = 0.0f;
        
        // Compute average using vectorized loads where possible
        #pragma unroll 4
        for (int d = d_start; d < d_end; ++d) {
            const int d_offset = d * in_h * in_w;
            #pragma unroll 4
            for (int h = h_start; h < h_end; ++h) {
                const int h_offset = h * in_w;
                #pragma unroll 4
                for (int w = w_start; w < w_end; ++w) {
                    sum += input[base_offset + d_offset + h_offset + w];
                }
            }
        }

        // Divide by full kernel volume (count_include_pad=True)
        const float kernel_vol = static_cast<float>(kernel_size * kernel_size * kernel_size);
        output[idx] = sum / kernel_vol;
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_d = input.size(2);
    const int in_h = input.size(3);
    const int in_w = input.size(4);
    
    const int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    const int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());
    
    const int total_elements = batch_size * channels * out_d * out_h * out_w;
    const int num_blocks = min((total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);
    
    avg_pool3d_forward_kernel_coalesced<<<num_blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with coalesced memory access");
}