#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// -----------------------------------------------------
// Optimized 1D convolution kernel with shared memory reduction
// Each output element is computed collaboratively by 64 threads.
// The reduction over the (group_size_in * K) elements is split among the 64 threads,
// then intra-warp reduction with __shfl_down_sync is used, and finally inter-warp
// partial sums are combined using shared memory.
// -----------------------------------------------------

__global__ void conv1d_forward_kernel_shared(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr, // can be null if no bias
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
    // Each output element is computed by TPO threads cooperatively
    const int TPO = 64;  // threads per output element

    // Determine the group id within the block and thread id within the group
    int group_id = threadIdx.x / TPO;          // which output element group in this block
    int thread_in_group = threadIdx.x % TPO;     // local index within the group

    // Total groups (i.e. output elements computed) per block
    int groups_per_block = blockDim.x / TPO;
    int out_index = blockIdx.x * groups_per_block + group_id;  // global output element index
    int total_outputs = N * C_out * L_out;
    if (out_index >= total_outputs) return;

    // Map the global output index to (n, out_ch, out_pos)
    int out_pos = out_index % L_out;
    int out_ch  = (out_index / L_out) % C_out;
    int n       = out_index / (L_out * C_out);

    // Determine group indices for input.
    int group_size_out = C_out / groups;
    int group_size_in  = C_in  / groups;
    int group_idx      = out_ch / group_size_out;

    // Each output element sums over "reduction_length" = group_size_in * K values
    float sum = 0.0f;
    int reduction_length = group_size_in * K;
    
    // Distribute reduction work among TPO threads
    for (int r = thread_in_group; r < reduction_length; r += TPO) {
        int local_in_ch = r / K;
        int k = r % K;
        int in_ch = group_idx * group_size_in + local_in_ch;
        int in_pos = out_pos * stride + k * dilation - padding;
        if (in_pos >= 0 && in_pos < L_in) {
            float x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
            float w_val = w[out_ch * (group_size_in * K) + local_in_ch * K + k];
            sum += x_val * w_val;
        }
    }

    // Use warp-level reduction for threads within each warp
    // Each group may span multiple warps if TPO > 32
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Allocate shared memory for inter-warp reduction within this group
    // There are (TPO/32) warp partial sums per group
    extern __shared__ float shared_data[]; // size allocated dynamically per block
    const int warps_per_group = TPO / 32;
    int lane = thread_in_group & 31;  // lane within the warp

    if (lane == 0) {
        int warp_id = thread_in_group / 32;
        shared_data[group_id * warps_per_group + warp_id] = sum;
    }
    __syncthreads();

    // Final reduction: let the first thread in each group sum the warp partial sums
    float total_sum = 0.0f;
    if (thread_in_group == 0) {
        for (int i = 0; i < warps_per_group; i++) {
            total_sum += shared_data[group_id * warps_per_group + i];
        }
        // Add bias if provided
        if (bias_ptr) {
            total_sum += bias_ptr[out_ch];
        }
        y[n * (C_out * L_out) + out_ch * L_out + out_pos] = total_sum;
    }
}

// -------------------------------------------------------
// conv1d forward implementation using the optimized kernel
// -------------------------------------------------------

at::Tensor conv1d_forward_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    // Check device and dtype
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    // x: [N, C_in, L_in]
    auto x_sizes = x.sizes();
    int64_t N    = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    // weight: [C_out, C_in/groups, K]
    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K     = w_sizes[2];

    // Calculate output length
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    // Create output tensor
    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    // Get bias pointer if provided
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Total number of output elements
    int total_outputs = N * C_out * L_out;

    // Launch configuration:
    // We use 64 threads per output element. For a block size of 256,
    // there are (256/64) = 4 output elements computed per block.
    int threads_per_block = 256;
    int groups_per_block = threads_per_block / 64;
    int gridSize = (total_outputs + groups_per_block - 1) / groups_per_block;

    // Shared memory size per block: each group contributes (TPO/32) floats
    size_t shared_mem_size = groups_per_block * (64 / 32) * sizeof(float); // groups_per_block * 2 * sizeof(float)

    conv1d_forward_kernel_shared<<<gridSize, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_shared failed: ", cudaGetErrorString(err));
    return y;
}

// -----------------------------------------------------
// Pybind11 binding with optional bias
// -----------------------------------------------------

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
        "Optimized 1D Convolution forward (CUDA) with shared memory reduction"
    );
}
