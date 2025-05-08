// Includes and namespace declarations
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// Warp-level reduction using shuffle instructions
__device__ inline float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Combined CUDA kernel for Conv Transpose 2D forward
// Each block computes one output pixel, and threads within the block partition the reduction
__global__ void conv_transpose2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
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
    // Map block ID to an output element
    int out_index = blockIdx.x;
    int ow = out_index % W_out;
    int oh = (out_index / W_out) % H_out;
    int oc = (out_index / (W_out * H_out)) % C_out;
    int n  = out_index / (W_out * H_out * C_out);

    float sum = 0.0f;

    // Partition the reduction domain (over input channels) among threads
    // For each input channel, loop over kernel height and width
    for (int ic = threadIdx.x; ic < C_in; ic += blockDim.x) {
        #pragma unroll
        for (int kh = 0; kh < kH; kh++) {
            #pragma unroll
            for (int kw = 0; kw < kW; kw++) {
                int i_val = oh + pH - kh;
                int j_val = ow + pW - kw;
                // Only contribute if the mapping aligns with the stride
                if ((i_val % sH) == 0 && (j_val % sW) == 0) {
                    int i_in = i_val / sH;
                    int j_in = j_val / sW;
                    if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                        int input_idx = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                        int weight_idx = ((ic * C_out + oc) * kH + kh) * kW + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Intra-warp reduction using warp shuffle
    sum = warpReduceSum(sum);

    // Shared memory reduction among warps in the block
    __shared__ float shared_sum[32];  // 32 is enough for up to 1024 threads (32 warps)
    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared_sum[warpId] = sum;
    }
    __syncthreads();

    // Final reduction by the first warp
    if (warpId == 0) {
        float blockSum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[lane] : 0.0f;
        blockSum = warpReduceSum(blockSum);
        if (lane == 0) {
            if (bias != nullptr) {
                blockSum += bias[oc];
            }
            int out_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
            output[out_idx] = blockSum;
        }
    }
}

// Host function wrapping the kernel launch
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
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

    // Calculate output dimensions for transposed convolution
    int H_out = (H_in - 1) * sH - 2 * pH + kH;
    int W_out = (W_in - 1) * sW - 2 * pW + kW;

    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    // Launch one block per output element
    int total_outputs = N * C_out * H_out * W_out;
    int threads = 256;

    conv_transpose2d_forward_kernel<<<total_outputs, threads>>>(
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

// Pybind11 module registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Optimized Conv Transpose 2D forward kernel combining unrolled loops and warp-level reduction",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
