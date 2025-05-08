#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp size constant
#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define SHARED_MEM_SIZE 2048

__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void conv_transposed_3d_cuda_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    const int C_out, const int D_out, const int H_out, const int W_out,
    const int kD, const int kH, const int kW,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int groups
) {
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    // Calculate output position
    const int total_elements = N * C_out * D_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (idx < total_elements) {
        const int w = idx % W_out;
        int tmp = idx / W_out;
        const int h = tmp % H_out;
        tmp /= H_out;
        const int d = tmp % D_out;
        tmp /= D_out;
        const int c_out = tmp % C_out;
        const int n = tmp / C_out;

        // Group information
        const int output_channels_per_group = C_out / groups;
        const int group = c_out / output_channels_per_group;
        const int c_out_in_group = c_out - group * output_channels_per_group;
        const int input_channels_per_group = C_in / groups;

        // Initialize shared memory for partial sums
        shared_mem[tid] = 0.0f;
        
        // Compute partial sums using shared memory
        for (int r = 0; r < kD; r++) {
            const int d_in_calc = d + pad_d - r;
            if (d_in_calc % stride_d != 0) continue;
            const int d_in = d_in_calc / stride_d;
            if (d_in < 0 || d_in >= D_in) continue;

            for (int s = 0; s < kH; s++) {
                const int h_in_calc = h + pad_h - s;
                if (h_in_calc % stride_h != 0) continue;
                const int h_in = h_in_calc / stride_h;
                if (h_in < 0 || h_in >= H_in) continue;

                for (int t = 0; t < kW; t++) {
                    const int w_in_calc = w + pad_w - t;
                    if (w_in_calc % stride_w != 0) continue;
                    const int w_in = w_in_calc / stride_w;
                    if (w_in < 0 || w_in >= W_in) continue;

                    float partial_sum = 0.0f;
                    #pragma unroll 4
                    for (int c = 0; c < input_channels_per_group; c++) {
                        const int actual_c_in = group * input_channels_per_group + c;
                        const int input_idx = (((n * C_in + actual_c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                        const int weight_idx = ((actual_c_in * output_channels_per_group + c_out_in_group) * (kD * kH * kW))
                                           + (r * kH * kW + s * kW + t);
                        
                        partial_sum += input[input_idx] * weight[weight_idx];
                    }
                    shared_mem[tid] += partial_sum;
                }
            }
        }
        
        __syncthreads();

        // Warp-level reduction
        float sum = shared_mem[tid];
        sum = warpReduceSum(sum);

        // First thread in warp writes result
        if (lane_id == 0) {
            float final_sum = sum;
            if (bias != nullptr) {
                final_sum += bias[c_out];
            }
            output[idx] = final_sum;
        }

        idx += blockDim.x * gridDim.x;
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

    const int out_pad_d = output_padding[0];
    const int out_pad_h = output_padding[1];
    const int out_pad_w = output_padding[2];

    const int D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + out_pad_d;
    const int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + out_pad_h;
    const int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + out_pad_w;

    const int output_channels_per_group = weight.size(1);
    const int C_out = output_channels_per_group * groups;

    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    const int total_elements = N * C_out * D_out * H_out * W_out;
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv_transposed_3d_cuda_kernel<<<grid, block, SHARED_MEM_SIZE * sizeof(float)>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward with shared memory optimization",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}