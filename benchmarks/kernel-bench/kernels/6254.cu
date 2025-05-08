#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that flattens the entire output index space and uses a grid-stride loop
// to evenly distribute work across threads and blocks.
__global__ void avg_pool3d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    int total_elements = batch_size * channels * out_d * out_h * out_w;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_total = blockDim.x * gridDim.x;

    while (tid < total_elements) {
        // Decompose linear index into (n, c, d_out, h_out, w_out)
        int tmp = tid;
        int w_out = tmp % out_w;
        tmp /= out_w;
        int h_out = tmp % out_h;
        tmp /= out_h;
        int d_out = tmp % out_d;
        tmp /= out_d;
        int c = tmp % channels;
        int n = tmp / channels;

        // Compute pooling window boundaries
        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;
        int d_end = d_start + kernel_size;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;

        // Clamp pooling window boundaries to input dimensions
        int d_start_clamped = d_start < 0 ? 0 : d_start;
        int h_start_clamped = h_start < 0 ? 0 : h_start;
        int w_start_clamped = w_start < 0 ? 0 : w_start;
        int d_end_clamped = d_end > in_d ? in_d : d_end;
        int h_end_clamped = h_end > in_h ? in_h : h_end;
        int w_end_clamped = w_end > in_w ? in_w : w_end;

        float sum = 0.0f;
        // Iterate over the pooling window
        for (int d = d_start_clamped; d < d_end_clamped; d++) {
            for (int h = h_start_clamped; h < h_end_clamped; h++) {
                // Compute starting index for the row in input
                int row_base = (((n * channels + c) * in_d + d) * in_h + h) * in_w;
                for (int w = w_start_clamped; w < w_end_clamped; w++) {
                    sum += input[row_base + w];
                }
            }
        }

        // Divide by full pooling volume (count_include_pad=True)
        int pool_volume = kernel_size * kernel_size * kernel_size;

    // Cache pool_volume for all iterations
    
    while (tid < total_elements) {
        // Decompose linear index into (n, c, d_out, h_out, w_out)
        int tmp = tid;
        int w_out = tmp % out_w;
        tmp /= out_w;
        int h_out = tmp % out_h;
        tmp /= out_h;
        int d_out = tmp % out_d;
        tmp /= out_d;
        int c = tmp % channels;
        int n = tmp / channels;

        // Compute pooling window boundaries
        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;
        int d_end = d_start + kernel_size;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;

        // Clamp pooling window boundaries to input dimensions
        int d_start_clamped = d_start < 0 ? 0 : d_start;
        int h_start_clamped = h_start < 0 ? 0 : h_start;
        int w_start_clamped = w_start < 0 ? 0 : w_start;
        int d_end_clamped = d_end > in_d ? in_d : d_end;
        int h_end_clamped = h_end > in_h ? in_h : h_end;
        int w_end_clamped = w_end > in_w ? in_w : w_end;

        float sum = 0.0f;

        // Precompute constant term for input offset
        int n_ch = n * channels + c;

        // Iterate over the pooling window with cached multiplications
        for (int d = d_start_clamped; d < d_end_clamped; d++) {
            int base_d = (n_ch * in_d + d) * in_h * in_w;
            for (int h = h_start_clamped; h < h_end_clamped; h++) {
                int base_row = base_d + h * in_w;
                for (int w = w_start_clamped; w < w_end_clamped; w++) {
                    sum += input[base_row + w];
                }
            }
        }

        // Divide by full pooling volume (count_include_pad=True)
        output[tid] = sum / static_cast<float>(pool_volume);

        tid += stride_total;
    }
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);

    // Compute output dimensions
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());
    int total_elements = batch_size * channels * out_d * out_h * out_w;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    avg_pool3d_forward_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with balanced workload distribution");
}
