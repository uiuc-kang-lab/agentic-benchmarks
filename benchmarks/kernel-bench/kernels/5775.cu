#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Device inline function for maximum
template <typename T>
__device__ __forceinline__ T warp_max(T a, T b) {
    return a > b ? a : b;
}

// Kernel: Each warp computes one output element using warp-level reduction
// Note: This kernel assumes that kernel_size * kernel_size <= 32
template <typename scalar_t>
__global__ void max_pool2d_kernel_warp_reduce(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    // Each warp processes one output element
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / 32;  // each warp has 32 threads
    int lane = threadIdx.x & 31;

    int total_outputs = batch_size * channels * output_height * output_width;
    if (warp_id < total_outputs) {
        // Map warp_id to output coordinates
        int ow = warp_id % output_width;
        int temp = warp_id / output_width;
        int oh = temp % output_height;
        temp /= output_height;
        int c = temp % channels;
        int b = temp / channels;

        // Total pooling elements = kernel_size * kernel_size (assumed <= 32)
        int pooling_elems = kernel_size * kernel_size;
        // Each lane loads one element if lane < pooling_elems, else it gets -infinity
        scalar_t thread_val = -std::numeric_limits<scalar_t>::infinity();
        if (lane < pooling_elems) {
            int kh = lane / kernel_size;
            int kw = lane % kernel_size;
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding + kw * dilation;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int input_idx = b * (channels * input_height * input_width) +
                                c * (input_height * input_width) +
                                ih * input_width + iw;
                thread_val = input[input_idx];
            }
        }
        
        // Warp-level reduction using __shfl_down_sync
        unsigned int mask = 0xFFFFFFFF;
        for (int offset = 16; offset > 0; offset /= 2) {
            scalar_t other = __shfl_down_sync(mask, thread_val, offset);
            thread_val = warp_max(thread_val, other);
        }

        // Lane 0 writes the maximum to output
        if (lane == 0) {
            output[warp_id] = thread_val;
        }
    }
}

// Forward function
torch::Tensor max_pool2d_cuda_forward_warp_reduce(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    // Compute output dimensions
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    // Ensure that our warp-level kernel is only used for small pooling windows
    TORCH_CHECK(kernel_size * kernel_size <= 32, "Warp-level reduction only supports kernel sizes with <= 32 elements.");

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Total number of output elements equals number of warps needed
    int total_outputs = batch_size * channels * output_height * output_width;
    // Each warp has 32 threads; choose a block size that is a multiple of 32
    const int threadsPerBlock = 128; // 4 warps per block
    int num_warps = (total_outputs + 0) ; // number of warps equals total_outputs
    int total_threads = num_warps * 32;
    int blocks = (total_threads + threadsPerBlock - 1) / threadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_warp_reduce", ([&] {
        max_pool2d_kernel_warp_reduce<scalar_t><<<blocks, threadsPerBlock>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward_warp_reduce, "Max Pool 2D forward with warp-level reduction (CUDA)");
}
