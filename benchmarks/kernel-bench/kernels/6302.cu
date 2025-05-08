#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size as a multiple of warp size (32)
constexpr int BLOCK_SIZE = 128;

// This kernel uses warp-level primitives (__shfl_down_sync) to perform the reduction
// across the pooling window for each output element. Each warp cooperatively computes
// one pooling result. The pooling window is flattened, and each thread in the warp
// sums a subset of the elements. A warp-level reduction then produces the final sum.

__global__ void avg_pool3d_forward_warp_reduce(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    // Total number of output elements
    int total_out = batch_size * channels * out_d * out_h * out_w;

    // Compute global thread id and derive warp id and lane id
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / 32; // one warp per output element
    int lane = global_tid & 31;    // equivalent to global_tid % 32

    // If warp_id exceeds the number of output elements, exit
    if (warp_id >= total_out) return;

    // Decompose warp_id into (n, c, d_out, h_out, w_out)
    int idx = warp_id;
    int w_out = idx % out_w;
    idx /= out_w;
    int h_out = idx % out_h;
    idx /= out_h;
    int d_out = idx % out_d;
    idx /= out_d;
    int c = idx % channels;
    int n = idx / channels;

    // Compute the starting indices of the pooling window
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    int kernel_volume = kernel_size * kernel_size * kernel_size;
    float sum = 0.0f;

    // Each thread in the warp processes a subset of the pooling window elements.
    // The pooling window is logically flattened into [0, kernel_volume).
    for (int m = lane; m < kernel_volume; m += 32) {
        // Map flat index m to 3D offsets (kd, kh, kw)
        int kd = m / (kernel_size * kernel_size);
        int rem = m % (kernel_size * kernel_size);
        int kh = rem / kernel_size;
        int kw = rem % kernel_size;

        int d = d_start + kd;
        int h = h_start + kh;
        int w = w_start + kw;

        float val = 0.0f;
        // Only include valid positions; padded regions contribute zero
        if (d >= 0 && d < in_d && h >= 0 && h < in_h && w >= 0 && w < in_w) {
            int input_index = (((n * channels + c) * in_d + d) * in_h + h) * in_w + w;
            val = input[input_index];
        }
        sum += val;
    }

    // Warp-level reduction using __shfl_down_sync to sum across all lanes
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Lane 0 writes the final averaged result to the output
    if (lane == 0) {
        output[warp_id] = sum / static_cast<float>(kernel_volume);
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

    // Calculate output dimensions
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    // Allocate output tensor
    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    // Total number of output elements. Each warp computes one output element.
    int total_out = batch_size * channels * out_d * out_h * out_w;
    // Total threads to launch = total_out * warpSize
    int total_threads = total_out * 32;
    int threads = BLOCK_SIZE;
    int blocks = (total_threads + threads - 1) / threads;

    avg_pool3d_forward_warp_reduce<<<blocks, threads>>>(
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
    m.def("forward", &forward, "3D Average Pooling forward with warp-level reduction (CUDA)");
}
