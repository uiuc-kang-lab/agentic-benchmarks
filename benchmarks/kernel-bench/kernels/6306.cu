#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;

// Device function to decompose linear index into output coordinates
__device__ __forceinline__ void decompose_index(
    int index, int out_w, int out_h, int out_d, int channels,
    int& n, int& c, int& d_out, int& h_out, int& w_out) {
    w_out = index % out_w;
    int tmp = index / out_w;
    h_out = tmp % out_h;
    tmp = tmp / out_h;
    d_out = tmp % out_d;
    tmp = tmp / out_d;
    c = tmp % channels;
    n = tmp / channels;
}

// Device function to compute pooling window boundaries
__device__ __forceinline__ void compute_window_bounds(
    int d_out, int h_out, int w_out,
    int stride, int padding, int kernel_size,
    int in_d, int in_h, int in_w,
    int& d_start, int& d_end, int& h_start, int& h_end, int& w_start, int& w_end) {
    
    d_start = max(d_out * stride - padding, 0);
    h_start = max(h_out * stride - padding, 0);
    w_start = max(w_out * stride - padding, 0);
    
    d_end = min(d_out * stride - padding + kernel_size, in_d);
    h_end = min(h_out * stride - padding + kernel_size, in_h);
    w_end = min(w_out * stride - padding + kernel_size, in_w);
}

// Device function to perform pooling operation
__device__ __forceinline__ float compute_pool_average(
    const float* input,
    int n, int c, 
    int d_start, int d_end,
    int h_start, int h_end,
    int w_start, int w_end,
    int in_d, int in_h, int in_w,
    int kernel_size) {
    
    float sum = 0.0f;
    int nc_offset = (n * channels + c) * in_d;
    
    for (int d = d_start; d < d_end; ++d) {
        int d_offset = (nc_offset + d) * in_h;
        for (int h = h_start; h < h_end; ++h) {
            int h_offset = (d_offset + h) * in_w;
            for (int w = w_start; w < w_end; ++w) {
                sum += input[h_offset + w];
            }
        }
    }
    
    return sum / (kernel_size * kernel_size * kernel_size);
}

__global__ void avg_pool3d_forward_modular(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_d * out_h * out_w;

    while (index < total_elements) {
        int n, c, d_out, h_out, w_out;
        decompose_index(index, out_w, out_h, out_d, channels, n, c, d_out, h_out, w_out);
        
        int d_start, d_end, h_start, h_end, w_start, w_end;
        compute_window_bounds(d_out, h_out, w_out, stride, padding, kernel_size,
                            in_d, in_h, in_w,
                            d_start, d_end, h_start, h_end, w_start, w_end);
        
        output[index] = compute_pool_average(input,
                                           n, c,
                                           d_start, d_end,
                                           h_start, h_end,
                                           w_start, w_end,
                                           in_d, in_h, in_w,
                                           kernel_size);
        
        index += blockDim.x * gridDim.x;
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    int out_d = (in_d + 2*padding - kernel_size)/stride + 1;
    int out_h = (in_h + 2*padding - kernel_size)/stride + 1;
    int out_w = (in_w + 2*padding - kernel_size)/stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());
    int total_elements = output.numel();

    dim3 blocks((total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);

    avg_pool3d_forward_modular<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling with modular device functions");
}