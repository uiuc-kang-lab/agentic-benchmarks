#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__global__ void conv_transpose2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kH, int kW,
    int sH, int sW,
    int pH, int pW
) {
    extern __shared__ float shared_mem[];
    float* block_sums = shared_mem;

    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (oh >= H_out || ow >= W_out) return;

    const int oc = blockIdx.z % C_out;
    const int n = blockIdx.z / C_out;
    
    float sum = 0.0f;

    for (int ic = 0; ic < C_in; ++ic) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                const int i_val = oh + pH - kh;
                const int j_val = ow + pW - kw;

                if ((i_val % sH == 0) && (j_val % sW == 0)) {
                    const int i_in = i_val / sH;
                    const int j_in = j_val / sW;

                    if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                        const int input_idx = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                        const int weight_idx = ((ic * C_out + oc) * kH + kh) * kW + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // First thread in warp stores partial sum
    if (threadIdx.x % 32 == 0)
        block_sums[threadIdx.y * blockDim.x + threadIdx.x / 32] = sum;

    __syncthreads();

    // Final accumulation with reduced atomic contention
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float final_sum = 0.0f;
        for (int i = 0; i < (blockDim.x * blockDim.y + 31) / 32; ++i)
            final_sum += block_sums[i];

        if (bias)
            final_sum += bias[oc];

        const int output_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
        output[output_idx] = final_sum;
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);
    
    const int C_out = weight.size(1);
    const int kH = weight.size(2);
    const int kW = weight.size(3);
    
    const int sH = stride[0];
    const int sW = stride[1];
    const int pH = padding[0];
    const int pW = padding[1];

    const int H_out = (H_in - 1) * sH - 2 * pH + kH;
    const int W_out = (W_in - 1) * sW - 2 * pW + kW;

    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    const float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        torch::Tensor bias = bias_obj.cast<torch::Tensor>();
        bias_ptr = bias.data_ptr<float>();
    }

    dim3 block(32, 8);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        N * C_out
    );

    const size_t shared_mem_size = (block.x * block.y / 32) * sizeof(float);
    
    conv_transpose2d_forward_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kH, kW,
        sH, sW,
        pH, pW
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2D optimized with shared memory reduction",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}