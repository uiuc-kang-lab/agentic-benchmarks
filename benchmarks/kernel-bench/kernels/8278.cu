#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// This kernel fuses shared memory based weight caching with vectorized output processing.
// Each block is responsible for one output channel and a portion of output positions processed in groups of 4.

__global__ void conv1d_forward_kernel_vectorized_shared(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr,
    float* __restrict__ y,
    const int N,
    const int C_in,
    const int L_in,
    const int C_out,
    const int K,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out
) {
    // Each block processes a fixed output channel and a given batch element.
    const int tid = threadIdx.x;
    const int out_ch = blockIdx.x;  // each block corresponds to one output channel
    const int batch_idx = blockIdx.z;  // batch index
    const int base_out_pos = (blockIdx.y * blockDim.x + tid) * 4;  // starting output position for this thread

    if (base_out_pos >= L_out) return;

    // Determine group info
    const int group_size_out = C_out / groups;
    const int group_size_in  = C_in / groups;
    const int group_idx      = out_ch / group_size_out;
    const int total_weights  = group_size_in * K;  // number of weights for one output channel

    // Allocate shared memory for the weight kernel of this output channel
    extern __shared__ float shmem[]; // size: group_size_in * K floats
    for (int i = tid; i < total_weights; i += blockDim.x) {
        shmem[i] = w[out_ch * total_weights + i];
    }
    __syncthreads();

    // Initialize vectorized accumulator for 4 output positions
    float4 output = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Loop over the input channels in the corresponding group
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        int in_ch = group_idx * group_size_in + local_in_ch;

        // Preload weights for this input channel from shared memory into a register cache
        // Assuming K is small (e.g., K <= 32). Adjust the size accordingly if needed.
        float weight_cache[32];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            weight_cache[k] = shmem[local_in_ch * K + k];
        }

        // Iterate over the kernel window positions
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float w_val = weight_cache[k];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int out_pos = base_out_pos + i;
                if (out_pos < L_out) {
                    int in_pos = out_pos * stride + k * dilation - padding;
                    if (in_pos >= 0 && in_pos < L_in) {
                        float x_val = x[batch_idx * (C_in * L_in) + in_ch * L_in + in_pos];
                        ((float*)&output)[i] += x_val * w_val;
                    }
                }
            }
        }
    }

    // Add bias if provided
    if (bias_ptr) {
        float bias_val = bias_ptr[out_ch];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            ((float*)&output)[i] += bias_val;
        }
    }

    // Write the computed output to global memory
    int out_offset = batch_idx * (C_out * L_out) + out_ch * L_out + base_out_pos;
    int remaining = min(4, L_out - base_out_pos);
    if (remaining == 4 && (((uintptr_t)(&y[out_offset])) & 15) == 0) {
        // Use vectorized store
        *((float4*)(&y[out_offset])) = output;
    } else {
        // Handle partial/unaligned stores
        for (int i = 0; i < remaining; ++i) {
            y[out_offset + i] = ((float*)&output)[i];
        }
    }
}

at::Tensor conv1d_forward_impl_vectorized_shared(
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

    auto x_sizes = x.sizes();
    int64_t N    = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K     = w_sizes[2];

    // Compute output length
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Configure grid and block dimensions for vectorized output processing
    const int threads_per_block = 128;
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(
        C_out,
        (L_out + (threads_per_block * 4) - 1) / (threads_per_block * 4),
        N
    );
    
    // Shared memory size: one block loads weights for a single output channel
    int group_size_in = C_in / groups;
    size_t sharedMemSize = group_size_in * K * sizeof(float);

    conv1d_forward_kernel_vectorized_shared<<<grid_dim, block_dim, sharedMemSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_vectorized_shared failed: ", cudaGetErrorString(err));
    return y;
}

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
            return conv1d_forward_impl_vectorized_shared(x, weight, bias, stride, padding, dilation, groups);
        },
        "Vectorized 1D Convolution forward with shared memory for weights (CUDA)"
    );
}
