#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// ----------------------------------------------------------------- 
// Hybrid 1D convolution CUDA kernel combining shared memory tiling and warp shuffle broadcast
// -----------------------------------------------------------------

// Each block computes one (n, out_ch) pair. Threads in a warp process different output positions.
// The weight tile for the output channel is first loaded into shared memory. Then, for each weight element,
// only the first lane of the warp reads from shared memory and uses __shfl_sync to broadcast the value to all lanes,
// reducing shared memory accesses and bank conflicts.

__global__ void conv1d_forward_kernel_hybrid(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr, // may be null if bias not provided
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
    // Each block handles one (n, out_ch) pair
    int n = blockIdx.x;
    int out_ch = blockIdx.y;

    // Compute warp lane id (assuming warp size of 32)
    const unsigned warpSize = 32;
    unsigned lane_id = threadIdx.x % warpSize;

    // Determine group info
    int group_size_out = C_out / groups;
    int group_size_in  = C_in / groups;
    int group_idx = out_ch / group_size_out;

    // Allocate shared memory for the weight tile corresponding to the output channel
    extern __shared__ float shw[]; // size: group_size_in * K floats
    int tile_size = group_size_in * K;

    // Load weights from global memory to shared memory in a coalesced manner
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        // Each output channel's weights tile is stored consecutively
        shw[i] = w[out_ch * tile_size + i];
    }
    __syncthreads();

    // Process multiple output positions per thread
    for (int out_pos = threadIdx.x; out_pos < L_out; out_pos += blockDim.x) {
        float sum = 0.0f;
        
        // Iterate over input channels within the group and over the kernel elements
        for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
            int in_ch = group_idx * group_size_in + local_in_ch;
            for (int k = 0; k < K; k++) {
                // Compute the corresponding input position
                int in_pos = out_pos * stride + k * dilation - padding;
                // Clamp the index and compute a binary mask (branchless)
                int clamped_in_pos = in_pos < 0 ? 0 : (in_pos >= L_in ? (L_in - 1) : in_pos);
                float mask = ((unsigned)in_pos < (unsigned)L_in) ? 1.0f : 0.0f;

                // Use warp-level shuffle to broadcast weight value loaded from shared memory
                // Only lane 0 performs the shared memory read
                float w_val = __shfl_sync(0xffffffff, shw[local_in_ch * K + k], 0);

                // Compute the index for input tensor x
                int x_index = n * (C_in * L_in) + in_ch * L_in + clamped_in_pos;
                sum += mask * x[x_index] * w_val;
            }
        }

        // Add bias if provided
        if (bias_ptr) {
            sum += bias_ptr[out_ch];
        }

        // Write the computed output
        int y_index = n * (C_out * L_out) + out_ch * L_out + out_pos;
        y[y_index] = sum;
    }
}

// -----------------------------------------------------------
// Hybrid conv1d forward implementation using the combined kernel
// -----------------------------------------------------------

at::Tensor conv1d_forward_impl_hybrid(
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

    // Get input dimensions: [N, C_in, L_in]
    auto x_sizes = x.sizes();
    int64_t N    = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    // Get weight dimensions: [C_out, C_in/groups, K]
    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K     = w_sizes[2];

    // Calculate output length
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Launch configuration: each block computes one (n, out_ch) pair
    dim3 grid(N, C_out);
    int threads = (L_out < 256) ? L_out : 256;

    // Shared memory size: weight tile of size group_size_in * K
    int group_size_in = C_in / groups;
    int sharedMem = group_size_in * K * sizeof(float);

    conv1d_forward_kernel_hybrid<<<grid, threads, sharedMem>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        (int)N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups, (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_hybrid failed: ", cudaGetErrorString(err));

    return y;
}

// -----------------------------------------------------
// Pybind11 binding for the hybrid convolution kernel
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
            return conv1d_forward_impl_hybrid(x, weight, bias, stride, padding, dilation, groups);
        },
        "Hybrid 1D Convolution forward (CUDA) combining shared memory tiling and warp shuffle broadcast"
    );
}
