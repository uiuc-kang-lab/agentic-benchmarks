#include <torch/extension.h>
#include <vector>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

__global__ void conv_transposed_3d_opt_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int totalElements,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int D_out, int H_out, int W_out,
    int groups
) {
    extern __shared__ float shmem[];
    const int tid = threadIdx.x;
    const int warpSize = 32;
    cg::thread_block blk = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(blk);

    // Each block processes multiple output elements
    for (int idx = blockIdx.x * blockDim.x + tid; idx < totalElements; idx += blockDim.x * gridDim.x) {
        int n = idx / (C_out * D_out * H_out * W_out);
        int residual = idx % (C_out * D_out * H_out * W_out);
        int c_out = residual / (D_out * H_out * W_out);
        residual %= (D_out * H_out * W_out);
        int d_out = residual / (H_out * W_out);
        residual %= (H_out * W_out);
        int h_out = residual / W_out;
        int w_out = residual % W_out;

        int output_channels_per_group = C_out / groups;
        int group = c_out / output_channels_per_group;
        int c_out_in_group = c_out - group * output_channels_per_group;
        int input_channels_per_group = C_in / groups;

        float acc = bias ? bias[c_out] : 0.0f;

        // Tile input/weights through shared memory
        const int shmem_size = (kD * kH * kW * input_channels_per_group) / warpSize;
        for (int t=0; t<(kD*kH*kW*input_channels_per_group + blockDim.x-1)/blockDim.x; ++t) {
            int linear_idx = t * blockDim.x + tid;
            if (linear_idx >= kD * kH * kW * input_channels_per_group) break;
            
            int r = linear_idx / (kH * kW * input_channels_per_group);
            linear_idx %= (kH * kW * input_channels_per_group);
            int s = linear_idx / (kW * input_channels_per_group);
            linear_idx %= (kW * input_channels_per_group);
            int t_k = linear_idx / input_channels_per_group;
            int c_in = linear_idx % input_channels_per_group;
            
            int d_in_calc = d_out * stride_d + r - pad_d;
            int h_in_calc = h_out + pad_h - s;
            int w_in_calc = w_out + pad_w - t_k;
            
            if (d_in_calc % stride_d != 0 || h_in_calc % stride_h != 0 || w_in_calc % stride_w != 0)
                continue;

            int d_in = d_in_calc / stride_d;
            int h_in = h_in_calc / stride_h;
            int w_in = w_in_calc / stride_w;

            if (d_in < 0 || d_in >= D_in || h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in)
                continue;

            int input_idx = (((n * C_in + (group * input_channels_per_group + c_in)) * D_in + d_in) * H_in + h_in) * W_in + w_in;
            int weight_idx = ((group * input_channels_per_group + c_in) * output_channels_per_group + c_out_in_group) * (kD * kH * kW) + (r * kH * kW + s * kW + t_k);

            shmem[tid % shmem_size] = input[input_idx] * weight[weight_idx];
            blk.sync();

            // Warp-level reduction
            float sum = shmem[tid % shmem_size];
            for (int offset = 16; offset > 0; offset /= 2)
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            
            if (tid % warpSize == 0)
                acc += sum;
        }

        if (tid % warpSize == 0)
            output[idx] = acc;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Maintain original dimension calculations
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    const int stride_d = stride[0];
    const int stride_h = stride[1];
    const int stride_w = stride[2];

    const int pad_d = padding[0];
    const int pad_h = padding[1];
    const int pad_w = padding[2];

    const int D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + output_padding[0];
    const int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + output_padding[1];
    const int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + output_padding[2];

    const int C_out = weight.size(1) * groups;
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    int totalElements = N * C_out * D_out * H_out * W_out;
    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;

    // Calculate shared memory size per block
    int input_channels_per_group = C_in / groups;
    int shmem_size = (kD * kH * kW * input_channels_per_group / 32) * sizeof(float);

    conv_transposed_3d_opt_kernel<<<gridSize, blockSize, shmem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        totalElements,
        N, C_in, D_in, H_in, W_in,
        C_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        D_out, H_out, W_out,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ConvTranspose3D with shared mem & warp reduce",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}