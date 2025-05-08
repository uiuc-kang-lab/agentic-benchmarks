#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

// This kernel assigns one block per output element.
// Each block performs a parallel reduction over its pooling window in shared memory,
// avoiding global atomic operations and thus minimizing contention.

template <typename scalar_t>
__global__ void parallel_maxpool3d_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    // Each block handles one output element, whose linear index is given by blockIdx.x
    int out_index = blockIdx.x;
    int w_out = out_index % output_w;
    int h_out = (out_index / output_w) % output_h;
    int d_out = (out_index / (output_w * output_h)) % output_d;
    int temp = out_index / (output_w * output_h * output_d);
    int c = temp % channels;
    int b = temp / channels;

    // Compute the top-left-front corner of the pooling window in the input tensor
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    // Determine valid pooling window bounds (handling boundaries)
    int k_d_min = (d_start < 0) ? ((-d_start + dilation - 1) / dilation) : 0;
    int k_d_max = min(kernel_size, (input_d - d_start + dilation - 1) / dilation);
    int k_h_min = (h_start < 0) ? ((-h_start + dilation - 1) / dilation) : 0;
    int k_h_max = min(kernel_size, (input_h - h_start + dilation - 1) / dilation);
    int k_w_min = (w_start < 0) ? ((-w_start + dilation - 1) / dilation) : 0;
    int k_w_max = min(kernel_size, (input_w - w_start + dilation - 1) / dilation);

    int count_d = k_d_max - k_d_min;
    int count_h = k_h_max - k_h_min;
    int count_w = k_w_max - k_w_min;
    int total_count = count_d * count_h * count_w;

    // Each thread processes a subset of the pooling window
    // Initialize local maximum value and corresponding input index
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    int local_max_idx = -1;

    for (int i = threadIdx.x; i < total_count; i += blockDim.x) {
        int kd_offset = i / (count_h * count_w);
        int rem = i % (count_h * count_w);
        int kh_offset = rem / count_w;
        int kw_offset = rem % count_w;
        int kd = k_d_min + kd_offset;
        int kh = k_h_min + kh_offset;
        int kw = k_w_min + kw_offset;

        int d_in = d_start + kd * dilation;
        int h_in = h_start + kh * dilation;
        int w_in = w_start + kw * dilation;
        int input_idx = (((b * channels + c) * input_d + d_in) * input_h + h_in) * input_w + w_in;
        scalar_t val = __ldg(&input[input_idx]);
        if (val > local_max) {
            local_max = val;
            local_max_idx = input_idx;
        }
    }

    // Allocate shared memory for reduction (shared memory avoids global atomics)
    extern __shared__ char smem[];
    scalar_t* sdata = (scalar_t*) smem;
    int* sindex = (int*)(smem + blockDim.x * sizeof(scalar_t));

    sdata[threadIdx.x] = local_max;
    sindex[threadIdx.x] = local_max_idx;
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x]) {
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
                sindex[threadIdx.x] = sindex[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes the final result to global memory; no global atomics are needed
    if (threadIdx.x == 0) {
        output[out_index] = sdata[0];
        if (indices != nullptr) {
            indices[out_index] = sindex[0];
        }
    }
}

// Host function to launch the kernel

torch::Tensor max_pool3d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices,
    bool ceil_mode) {

    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int channels = input_sizes[1];
    int input_d = input_sizes[2];
    int input_h = input_sizes[3];
    int input_w = input_sizes[4];

    // Compute output dimensions
    float d_out_f = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;
    float h_out_f = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;
    float w_out_f = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;
    int output_d = ceil_mode ? std::ceil(d_out_f) : std::floor(d_out_f);
    int output_h = ceil_mode ? std::ceil(h_out_f) : std::floor(h_out_f);
    int output_w = ceil_mode ? std::ceil(w_out_f) : std::floor(w_out_f);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ?
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    // Total number of output elements
    int total_output = batch_size * channels * output_d * output_h * output_w;
    int threads = 128;  // Threads per block for parallel reduction
    dim3 grid(total_output);
    size_t shared_mem_size = threads * (sizeof(float) + sizeof(int));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda", ([&] {
        parallel_maxpool3d_shared_kernel<scalar_t><<<grid, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size, channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size, stride, padding, dilation);
    }));

    if (return_indices) {
        return torch::stack({output, indices}, 0);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool3d_cuda_forward, "3D Max Pool forward with block-level shared memory reduction (CUDA)");
}
