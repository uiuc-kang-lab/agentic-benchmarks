#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized transposed convolution kernel using shared memory and warp-level reduction
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding) {

    // Each block is responsible for computing one output element at (b, oc, h, w)
    int out_index = blockIdx.x;
    int w = out_index % W_out;
    out_index /= W_out;
    int h = out_index % H_out;
    out_index /= H_out;
    int oc = out_index % C_out;
    int b = out_index / C_out;

    float sum = 0.0f;
    // Total iterations over input channel and kernel area
    int total_iters = C_in * K * K;

    // Each thread computes a partial sum over a subset of positions
    for (int pos = threadIdx.x; pos < total_iters; pos += blockDim.x) {
         int ic = pos / (K * K);
         int rem = pos % (K * K);
         int kh = rem / K;
         int kw = rem % K;

         // Compute candidate input indices (reverse relation for transposed convolution)
         int h_in_candidate = h + padding - kh;
         int w_in_candidate = w + padding - kw;
         if ((h_in_candidate % stride == 0) && (w_in_candidate % stride == 0)) {
             int h_in = h_in_candidate / stride;
             int w_in = w_in_candidate / stride;
             if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                 int input_idx = b * (C_in * H_in * W_in) + ic * (H_in * W_in) + h_in * W_in + w_in;
                 int weight_idx = ic * (C_out * K * K) + oc * (K * K) + kh * K + kw;
                 sum += input[input_idx] * weight[weight_idx];
             }
         }
    }

    // First-stage reduction within a warp using warp shuffle
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
         sum += __shfl_down_sync(mask, sum, offset);
    }

    // Allocate shared memory dynamically for inter-warp reduction
    extern __shared__ float sdata[];
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
         sdata[warpId] = sum;
    }
    __syncthreads();

    // Second-stage reduction across warp results. Assumes number of warps per block is <= warpSize.
    int nWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < nWarps) {
         float blockSum = sdata[threadIdx.x];
         for (int offset = warpSize / 2; offset > 0; offset /= 2) {
              blockSum += __shfl_down_sync(0xffffffff, blockSum, offset);
         }
         if (threadIdx.x == 0) {
              if (bias != nullptr) {
                   blockSum += bias[oc];
              }
              int out_offset = b * (C_out * H_out * W_out) + oc * (H_out * W_out) + h * W_out + w;
              output[out_offset] = blockSum;
         }
    }
}

// Forward function that wraps the optimized CUDA kernel
torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    if (bias.has_value()) {
         TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
         TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    // This optimized kernel supports only groups=1 and output_padding==0 currently
    TORCH_CHECK(groups == 1, "Only groups==1 is supported in the optimized kernel");
    TORCH_CHECK(output_padding == 0, "Only output_padding==0 is supported in the optimized kernel");

    int B = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    // Assuming weight has shape (C_in, C_out, K, K) for square kernel
    int K = weight.size(2);
    int C_out = weight.size(1);
    int H_out = (H_in - 1) * stride - 2 * padding + K;
    int W_out = (W_in - 1) * stride - 2 * padding + K;

    auto output_tensor = torch::zeros({B, C_out, H_out, W_out}, input.options());

    // Launch one block per output element
    int total_outputs = (B * C_out * H_out * W_out + 255) / 256;
    int threads = 256; // Number of threads per block
    int numWarps = (threads + 31) / 32;
    int shared_mem_bytes = numWarps * sizeof(float);

    conv_transpose2d_kernel<<<total_outputs, threads, shared_mem_bytes>>>(
         input.data_ptr<float>(),
         weight.data_ptr<float>(),
         bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
         output_tensor.data_ptr<float>(),
         B, C_in, H_in, W_in,
         C_out, H_out, W_out,
         K, stride, padding);
    
    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Optimized ConvTranspose2d forward (CUDA) using shared memory and warp-level reduction");
}
