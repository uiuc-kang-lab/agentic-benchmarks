#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Helper device function to compute minimum
__device__ __forceinline__ int dev_min(int a, int b) {
    return a < b ? a : b;
}

// Each block computes one output element using block-level reduction in shared memory
// This design avoids global atomic operations by using intra-block synchronization

template <typename scalar_t>
__global__ void max_pool3d_forward_block_kernel(
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

    // Each block is responsible for one output element
    int out_idx = blockIdx.x;
    int total_outputs = batch_size * channels * output_d * output_h * output_w;
    if (out_idx >= total_outputs) return;

    // Decode linear index to (b, c, d_out, h_out, w_out)
    int w_out = out_idx % output_w;
    int h_out = (out_idx / output_w) % output_h;
    int d_out = (out_idx / (output_w * output_h)) % output_d;
    int c = (out_idx / (output_w * output_h * output_d)) % channels;
    int b = out_idx / (output_w * output_h * output_d * channels);

    // Compute starting positions in the input tensor
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    // Compute valid kernel bounds to avoid out-of-bound accesses
    int k_d_start = (d_start < 0) ? ((-d_start + dilation - 1) / dilation) : 0;
    int k_d_end = dev_min(kernel_size, (input_d - d_start + dilation - 1) / dilation);

    int k_h_start = (h_start < 0) ? ((-h_start + dilation - 1) / dilation) : 0;
    int k_h_end = dev_min(kernel_size, (input_h - h_start + dilation - 1) / dilation);

    int k_w_start = (w_start < 0) ? ((-w_start + dilation - 1) / dilation) : 0;
    int k_w_end = dev_min(kernel_size, (input_w - w_start + dilation - 1) / dilation);

    int pool_size = (k_d_end - k_d_start) * (k_h_end - k_h_start) * (k_w_end - k_w_start);

    // Each thread in the block processes a subset of the pooling region
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    int64_t thread_max_idx = -1;

    for (int i = threadIdx.x; i < pool_size; i += blockDim.x) {
        // Map flattened index to (k_d, k_h, k_w)
        int npool = (k_h_end - k_h_start) * (k_w_end - k_w_start);
        int kd = i / npool;
        int rem = i % npool;
        int kh = rem / (k_w_end - k_w_start);
        int kw = rem % (k_w_end - k_w_start);

        int d_in = d_start + (k_d_start + kd) * dilation;
        int h_in = h_start + (k_h_start + kh) * dilation;
        int w_in = w_start + (k_w_start + kw) * dilation;

        int64_t input_idx = (((int64_t)b * channels + c) * input_d + d_in) * input_h * input_w +
                            h_in * input_w + w_in;
        scalar_t val = input[input_idx];
        if (val > thread_max) {
            thread_max = val;
            thread_max_idx = input_idx;
        }
    }

    // Allocate shared memory for reduction
    extern __shared__ char smem[];
    scalar_t* sdata_val = reinterpret_cast<scalar_t*>(smem);
    int64_t* sdata_idx = reinterpret_cast<int64_t*>(sdata_val + blockDim.x);

    sdata_val[threadIdx.x] = thread_max;
    sdata_idx[threadIdx.x] = thread_max_idx;
    __syncthreads();

    // Intra-block reduction to compute the maximum value and its index
    for (unsigned int stride_val = blockDim.x / 2; stride_val > 0; stride_val /= 2) {
        if (threadIdx.x < stride_val) {
            if (sdata_val[threadIdx.x + stride_val] > sdata_val[threadIdx.x]) {
                sdata_val[threadIdx.x] = sdata_val[threadIdx.x + stride_val];
                sdata_idx[threadIdx.x] = sdata_idx[threadIdx.x + stride_val];
            }
        }
        __syncthreads();
    }

    // Write the result for this output element
    if (threadIdx.x == 0) {
        output[out_idx] = sdata_val[0];
        if (indices != nullptr) {
            indices[out_idx] = sdata_idx[0];
        }
    }
}


// Host function that sets up and launches the kernel

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

    float d_out_f = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;
    float h_out_f = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;
    float w_out_f = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;

    int output_d = ceil_mode ? int(ceil(d_out_f)) : int(floor(d_out_f));
    int output_h = ceil_mode ? int(ceil(h_out_f)) : int(floor(h_out_f));
    int output_w = ceil_mode ? int(ceil(w_out_f)) : int(floor(w_out_f));

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong));
    }

    int total_outputs = batch_size * channels * output_d * output_h * output_w;
    const int threads = 128;
    const int blocks = total_outputs;  // One block per output element

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda", ([&] {
        size_t smem_bytes = threads * (sizeof(scalar_t) + sizeof(int64_t));
        max_pool3d_forward_block_kernel<scalar_t><<<blocks, threads, smem_bytes>>>(
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
    m.def("forward", &max_pool3d_cuda_forward, "Max Pool 3D forward block-level reduction (CUDA)");
}
