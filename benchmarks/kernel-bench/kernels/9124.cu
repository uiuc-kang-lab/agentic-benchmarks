#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// Constant memory for weights
__constant__ float c_weight[16384];

// Warp-level reduction using shuffle instructions
__device__ inline float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel with inline warp reduction and shared memory
__global__ void conv_transpose2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int H_out,
    int W_out,
    int kH,
    int kW,
    int sH,
    int sW,
    int pH,
    int pW
) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // Each block computes one output pixel, one per thread
    int ow = bid % W_out;
    int oh = (bid / W_out) % H_out;
    int oc = (bid / (W_out * H_out)) % C_out;
    int n  = bid / (W_out * H_out * C_out);

    int L = C_in * kH * kW;
    float local_sum = 0.0f;

    for (int r = tid; r < L; r += blockDim.x) {
        int ic = r / (kH * kW);
        int rem = r % (kH * kW);
        int kh = rem / kW;
        int kw = rem % kW;
        int i_val = oh + pH - kh;
        int j_val = ow + pW - kw;

        if ((i_val % sH == 0) && (j_val % sW == 0)) {
            int i_in = i_val / sH;
            int j_in = j_val / sW;
            if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                int input_idx = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                int weight_idx = ((ic * C_out + oc) * kH + kh) * kW + kw;
                local_sum += input[input_idx] * c_weight[weight_idx];
            }
        }
    }

    local_sum = warpReduceSum(local_sum);

    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) shared_mem[warp_id] = local_sum;
    __syncthreads();

    local_sum = (tid < (blockDim.x + 31) / 32 ? shared_mem[lane] : 0);
    if (warp_id == 0) {
        local_sum = warpReduceSum(local_sum);
        if (lane == 0) {
            if (bias != nullptr) local_sum += bias[oc];
            int output_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
            output[output_idx] = local_sum;
        }
    }
}

// Host function
void conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    int weight_size = weight.numel() * sizeof(float);
    if (weight_size > 64 * 1024) {
        c10::optional<torch::Tensor> bias = c10::nullopt;
        if (!bias_obj.is_none()) {
            bias = bias_obj.cast<torch::Tensor>();
        }
        return at::conv_transpose2d(x, weight, bias, stride, padding);
    }

    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), weight_size);

    torch::Tensor bias;
    const float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
        bias_ptr = bias.data_ptr<float>();
    }

    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int C_out = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);
    int sH = stride[0];
    int sW = stride[1];
    int pH = padding[0];
    int pW = padding[1];

    int H_out = (H_in - 1) * sH - 2 * pH + kH;
    int W_out = (W_in - 1) * sW - 2 * pW + kW;

    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    int total_outputs = N * C_out * H_out * W_out;
    int threads = 256;

    conv_transpose2d_forward_kernel<<<total_outputs, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
        N, C_in, H_in, W_in, C_out, H_out, W_out,
        kH, kW, sH, sW, pH, pW
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with shared memory optimizations",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
