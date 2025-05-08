#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// ---------------------------------------------------------------------
// Optimized 1D convolution kernel using block-level reduction
// and atomicAdd only for final accumulation of partial results.
// Each output element (indexed by n, out_ch, out_pos) is computed
// by partitioning the input channels (per group) into chunks.
// Only the block handling the first chunk adds the bias (if provided).
// The final atomicAdd per block minimizes global memory contention.
// ---------------------------------------------------------------------

__global__ void conv1d_forward_atomic_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr, // may be nullptr if no bias
    float* __restrict__ y,
    const int N,         // batch size
    const int C_in,      // input channels
    const int L_in,      // input length
    const int C_out,     // output channels
    const int K,         // kernel size
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out      // output length
) {
    // Each output element is computed by summing over a chunk of the input channels.
    // Compute number of input channels per group
    int local_channels = C_in / groups;  // channels per group
    // Divide the local channels into chunks of size blockDim.x
    int num_chunks = (local_channels + blockDim.x - 1) / blockDim.x;

    // Determine which output element and which chunk this block is responsible for
    int global_block_id = blockIdx.x;
    int out_elem = global_block_id / num_chunks; // flatted index for (n, out_ch, out_pos)
    int chunk_idx = global_block_id % num_chunks;  // which chunk of local channels

    // Decode out_elem into (n, out_ch, out_pos)
    int out_pos = out_elem % L_out;
    int temp = out_elem / L_out;
    int out_ch = temp % C_out;
    int n = temp / C_out;

    // Determine the corresponding input group for this output channel
    int group_size_out = C_out / groups;
    int group_idx = out_ch / group_size_out;

    // Determine the range of local channel indices handled by this block
    int chunk_start = chunk_idx * blockDim.x;
    int chunk_end = (chunk_start + blockDim.x < local_channels) ? (chunk_start + blockDim.x) : local_channels;

    float partial_sum = 0.0f;
    // Each thread in the block processes a subset of channels in this chunk
    for (int local_in_ch = chunk_start + threadIdx.x; local_in_ch < chunk_end; local_in_ch += blockDim.x) {
        // Map local channel index to global input channel
        int in_ch = group_idx * local_channels + local_in_ch;
        // Loop over the kernel width
        for (int k = 0; k < K; k++) {
            int in_pos = out_pos * stride + k * dilation - padding;
            if (in_pos >= 0 && in_pos < L_in) {
                float x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                float w_val = w[out_ch * (local_channels * K) + local_in_ch * K + k];
                partial_sum += x_val * w_val;
            }
        }
    }

    // Block-level reduction using shared memory
    extern __shared__ float sdata[]; // dynamically allocated shared memory
    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // Reduce within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The first thread of the block accumulates the partial sum into the output
    if (threadIdx.x == 0) {
        float block_sum = sdata[0];
        // Only add bias once per output element (in the block handling the first chunk)
        if (bias_ptr != nullptr && chunk_idx == 0) {
            block_sum += bias_ptr[out_ch];
        }
        // Atomic add to global output to avoid race conditions across blocks
        atomicAdd(&y[n * (C_out * L_out) + out_ch * L_out + out_pos], block_sum);
    }
}

// ---------------------------------------------------------------------
// Host implementation of the optimized conv1d forward using atomic accumulation
// ---------------------------------------------------------------------

at::Tensor conv1d_forward_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    // x shape: [N, C_in, L_in]
    auto x_sizes = x.sizes();
    int64_t N    = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    // weight shape: [C_out, C_in/groups, K]
    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K     = w_sizes[2];

    // Calculate output length for conv1d
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    // Allocate output tensor and initialize to 0 for safe atomic accumulation
    auto y = torch::zeros({N, C_out, L_out}, x.options().dtype(at::kFloat));

    // Bias handling (if provided)
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Determine tiling along the input channel dimension per group
    int local_channels = C_in / groups;
    int block_size = 256; // using 256 threads per block
    // Each block processes a chunk of input channels of size equal to block_size
    int num_chunks = (local_channels + block_size - 1) / block_size;

    // Each output element (n, out_ch, out_pos) is computed in num_chunks blocks
    int total_out_elements = N * C_out * L_out;
    int grid_size = total_out_elements * num_chunks;

    // Launch the kernel with dynamic shared memory size
    conv1d_forward_atomic_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        (int)N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups, (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_atomic_kernel failed: ", cudaGetErrorString(err));

    return y;
}

// ---------------------------------------------------------------------
// Pybind11 binding to expose the forward method
// ---------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](at::Tensor x,
           at::Tensor weight,
           py::object bias_obj,
           int64_t stride,
           int64_t padding,
           int64_t dilation,
           int64_t groups) {
            c10::optional<at::Tensor> bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<at::Tensor>();
            }
            return conv1d_forward_impl(x, weight, bias, stride, padding, dilation, groups);
        },
        "Optimized 1D Convolution forward (CUDA) with minimal atomic operations"
    );
}
