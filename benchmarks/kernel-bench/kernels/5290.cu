#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARPSIZE = 32;
constexpr int BLOCK_FEATURES = 128;

__device__ void warp_reduce_max(float& val, int& idx) {
    for (int offset = WARPSIZE/2; offset > 0; offset /= 2) {
        float tmp_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int tmp_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (tmp_val > val) {
            val = tmp_val;
            idx = tmp_idx;
        }
    }
}

__global__ void max_pool1d_sharedmem_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int output_length,
    bool return_indices)
{
    __shared__ float smem_vals[BLOCK_FEATURES][WARPSIZE];
    __shared__ int smem_idxs[BLOCK_FEATURES][WARPSIZE];

    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    const int i = blockIdx.x * WARPSIZE + threadIdx.x;
    
    if (b >= batch_size || c >= num_channels || i >= output_length) return;

    const int input_start = i * stride - padding;
    float max_val = -FLT_MAX;
    int max_idx = -1;

    for (int k = threadIdx.y; k < kernel_size; k += blockDim.y) {
        const int pos = input_start + k * dilation;
        if (pos >= 0 && pos < input_length) {
            const float val = input[b * num_channels * input_length + c * input_length + pos];
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }

    smem_vals[threadIdx.y][threadIdx.x] = max_val;
    smem_idxs[threadIdx.y][threadIdx.x] = max_idx;
    __syncthreads();

    if (threadIdx.y == 0) {
        float final_max = smem_vals[threadIdx.x][threadIdx.x];
        int final_idx = smem_idxs[threadIdx.x][threadIdx.x];
        for (int f = 1; f < blockDim.y; ++f) {
            if (smem_vals[f][threadIdx.x] > final_max) {
                final_max = smem_vals[f][threadIdx.x];
                final_idx = smem_idxs[f][threadIdx.x];
            }
        }
        warp_reduce_max(final_max, final_idx);

        if (threadIdx.x == 0) {
            const int out_idx = b * num_channels * output_length + c * output_length + i/WARPSIZE;
            output[out_idx] = final_max;
            if (return_indices) indices[out_idx] = final_idx;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices)
{
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    const int batch_size = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);

    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;

    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, 
            options.dtype(torch::kInt64));
    }

    const dim3 blocks(
        (output_length + WARPSIZE - 1) / WARPSIZE,
        (num_channels + BLOCK_FEATURES - 1) / BLOCK_FEATURES,
        batch_size
    );
    const dim3 threads(WARPSIZE, BLOCK_FEATURES);

    max_pool1d_sharedmem_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        batch_size,
        num_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        dilation,
        output_length,
        return_indices
    );

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward with shared memory optimizations (CUDA)");
}
