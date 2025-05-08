#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace py = pybind11;

// This kernel computes one output element per block using a two-level reduction:
// Each thread in the block computes partial sums over a subset of the reduction domain (in_channels * kernel_h * kernel_w).
// Then, warp-level reduction with __shfl_down_sync() is used to reduce within each warp, and the warp leaders write
// their sums to shared memory. Finally, a lightweight reduction among the warp sums produces the final output value.

// Kernel: each block processes one output pixel.
// The output tensor shape is: [N, out_channels, out_h, out_w].

__global__ void conv_transpose2d_block_reduce_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    bool has_bias
) {
    // Each block computes one output element
    int out_index = blockIdx.x;
    // Decode out_index into (n, oc, out_y, out_x)
    int tmp = out_index;
    int out_x = tmp % out_w;
    tmp /= out_w;
    int out_y = tmp % out_h;
    tmp /= out_h;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    // Total reduction domain size
    int reduction_size = in_channels * kernel_h * kernel_w;

    // Two-level reduction variables with bank-conflict-free shared memory layout
    // Use a two-level reduction: first, warp-level reduction using shuffle, then block-level reduction among warp results.
    // Add padding to avoid bank conflicts (32 banks on modern GPUs)
    extern __shared__ float shared[];
    
    // Pad shared memory array to avoid bank conflicts
    #define BANK_SIZE 32
    #define PADDED_SIZE(n) (((n) + (BANK_SIZE - 1)) & ~(BANK_SIZE - 1))
    
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    
    // Pre-compute warp index with padding to avoid bank conflicts
    int warp_idx = tid >> 5;
    int padded_warp_idx = PADDED_SIZE(warp_idx);

    // Precompute base coordinates for output pixel
    int base_y = out_y + pad_h;
    int base_x = out_x + pad_w;

    // Loop over reduction domain: each element corresponds to a combination of (ic, ky, kx)
    // We partition the reduction among threads in the block
    for (int r = tid; r < reduction_size; r += blockDim.x) {
        int ic = r / (kernel_h * kernel_w);
        int rem = r % (kernel_h * kernel_w);
        int ky = rem / kernel_w;
        int kx = rem % kernel_w;

        // Compute the corresponding input coordinate
        int t_y = base_y - ky;
        int t_x = base_x - kx;
        // Check alignment with stride
        if ((t_y % stride_h) != 0 || (t_x % stride_w) != 0) continue;
        int in_y = t_y / stride_h;
        int in_x = t_x / stride_w;
        if (in_y < 0 || in_y >= in_h || in_x < 0 || in_x >= in_w) continue;

        // Compute indices in input and weight tensors
        int input_idx = ((n * in_channels + ic) * in_h + in_y) * in_w + in_x;
        int weight_idx = ((ic * out_channels + oc) * kernel_h + ky) * kernel_w + kx;

        local_sum += input[input_idx] * weight[weight_idx];
    }

    // --- First level reduction: warp-level using shuffle ---
    // Each warp reduces its own partial sum.
    unsigned int mask = 0xffffffff;
    int lane = tid & 31;  // lane id within the warp
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Warp leader writes its sum to shared memory
    if (lane == 0) {
        shared[tid >> 5] = local_sum;  // tid/32
    }
    __syncthreads();

    // --- Second level reduction: reduce warp-level results ---
    // Let the first warp reduce the warp-level sums. Number of warps per block = blockDim.x/32
    int num_warps = blockDim.x >> 5;
    float final_sum = 0.0f;
    if (tid < num_warps) {
        final_sum = shared[tid];
    } else {
        final_sum = 0.0f;
    }
    __syncthreads();
    if (tid < 32) {
        // Reduce the values in the first warp using shuffle
        for (int offset = 16; offset > 0; offset /= 2) {
            final_sum += __shfl_down_sync(mask, final_sum, offset);
        }
    }

    // Thread 0 in block writes the final result
    if (tid == 0) {
        float result = (has_bias ? bias[oc] : 0.0f) + final_sum;
        // Compute the final output index with shape [N, out_channels, out_h, out_w]
        int out_tensor_idx = ((n * out_channels + oc) * out_h + out_y) * out_w + out_x;
        output[out_tensor_idx] = result;
    }
}

// Host function to launch the kernel

torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    // Input shape: [N, in_channels, in_h, in_w]
    int N = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);

    // Weight shape: [in_channels, out_channels, kernel_h, kernel_w]
    int out_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];

    // Output dimensions for transposed convolution
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w;

    auto output = torch::zeros({N, out_channels, out_h, out_w}, input.options());

    // Total number of output elements (each computed by one block)
    int total_outputs = N * out_channels * out_h * out_w;

    // Choose block size (number of threads per block). For reduction, 256 threads is a common choice.
    int threads = 256;
    dim3 blocks(total_outputs);

    // Shared memory size: one float per warp, number of warps = threads/32
    size_t shared_mem_size = (threads / 32) * sizeof(float);

    bool has_bias = (bias_opt.has_value() && bias_opt.value().numel() > 0);
    const float* bias_ptr = has_bias ? bias_opt.value().data_ptr<float>() : nullptr;

    // Launch the kernel using the current CUDA stream
    conv_transpose2d_block_reduce_kernel<<<blocks, threads, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, in_channels, in_h, in_w,
        out_channels, out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        has_bias
    );

    return output;
}

// Pybind11 binding

torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    c10::optional<torch::Tensor> bias = c10::nullopt;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
    }
    return conv_transpose2d_forward_cuda(input, weight, bias, stride, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward (shared memory + warp-level reduction)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
