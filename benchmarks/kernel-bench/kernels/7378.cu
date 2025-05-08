#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel that computes one output element per block. Threads in the block cooperate to perform the required reduction using shared memory and warp-level primitives.

__global__ void conv2d_cuda_kernel_shared_reduce(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    // Each block computes one output element
    int out_index = blockIdx.x;
    int total_out = N * C_out * H_out * W_out;
    if (out_index >= total_out) return;

    // Decode output element indices: (n, c_out, h_out, w_out)
    int w_out = out_index % W_out;
    int tmp = out_index / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    // Calculate the reduction size: over the contributing input channels in the group and kernel spatial dimensions
    int per_group = C_in / groups; // number of input channels per group
    int reduction_size = per_group * K_h * K_w;

    // Each thread in the block will compute a partial sum over a subset of the reduction elements
    float partial_sum = 0.0f;
    for (int i = threadIdx.x; i < reduction_size; i += blockDim.x) {
        int c_in_local = i / (K_h * K_w); // local channel index within the group
        int rem = i % (K_h * K_w);
        int k_h = rem / K_w;
        int k_w = rem % K_w;
        int c_in = (c_out / (C_out / groups)) * per_group + c_in_local;

        int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
        int w_in = w_out * stride_w - padding_w + k_w * dilation_w;
        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
            int weight_idx = (((c_out * per_group + c_in_local) * K_h + k_h) * K_w) + k_w;
            partial_sum += input[input_idx] * weight[weight_idx];
        }
    }

    // Intra-warp reduction using warp-level primitives
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    // Allocate shared memory for warp-level sums
    extern __shared__ float shmem[];
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Write the reduced value of each warp to shared memory
    if (lane_id == 0) {
        shmem[warp_id] = partial_sum;
    }
    __syncthreads();

    // Let the first warp reduce the warp sums
    float final_sum = 0.0f;
    int num_warps = (blockDim.x + 31) / 32;
    if (threadIdx.x < num_warps) {
        final_sum = shmem[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            final_sum += __shfl_down_sync(mask, final_sum, offset);
        }
    }

    // Thread 0 writes the final sum to output (adding bias if provided)
    if (threadIdx.x == 0) {
        if (bias != nullptr) {
            final_sum += bias[c_out];
        }
        int out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
        output[out_idx] = final_sum;
    }
}

// C++ interface

torch::Tensor conv2d_cuda_shared_reduce(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    input = input.contiguous();
    weight = weight.contiguous();

    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    int64_t C_out = weight.size(0);
    int64_t K_h = weight.size(2);
    int64_t K_w = weight.size(3);

    int64_t stride_h = stride[0];
    int64_t stride_w = stride[1];
    int64_t padding_h = padding[0];
    int64_t padding_w = padding[1];
    int64_t dilation_h = dilation[0];
    int64_t dilation_w = dilation[1];

    int64_t H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int64_t W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    int total_outputs = N * C_out * H_out * W_out;
    // Choose a block size (e.g., 128 threads per block) for the reduction over each output element.
    const int threads_per_block = 128;
    // Shared memory size: one float per warp in the block
    int shared_mem_size = ((threads_per_block + 31) / 32) * sizeof(float);

    conv2d_cuda_kernel_shared_reduce<<<total_outputs, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_opt.has_value() ? bias_opt.value().contiguous().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in conv2d_cuda_kernel_shared_reduce: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_shared_reduce, "Convolution with shared memory reduction optimized (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}
