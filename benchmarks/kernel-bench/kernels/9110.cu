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

// Combined kernel: uses grid-stride loops to allow each block to process multiple output pixels
// and uses warp-level reduction with shared memory to efficiently reduce over the large reduction domain.
__global__ void conv_transpose2d_forward_kernel_combined(
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
    // Total number of output pixels
    int total_outputs = N * C_out * H_out * W_out;
    
    // Use a grid-stride loop so each block can process multiple outputs if needed
    for (int out_idx = blockIdx.x; out_idx < total_outputs; out_idx += gridDim.x) {
        // Decode the linear index into (n, oc, oh, ow)
        int ow = out_idx % W_out;
        int temp = out_idx / W_out;
        int oh = temp % H_out;
        temp /= H_out;
        int oc = temp % C_out;
        int n = temp / C_out;
        
        float local_sum = 0.0f;
        // Reduction domain: over all input channels and kernel positions
        int L = C_in * kH * kW;
        
        // Each thread in the block sums over a slice of the reduction domain
        for (int r = threadIdx.x; r < L; r += blockDim.x) {
            int ic = r / (kH * kW);
            int rem = r % (kH * kW);
            int kh = rem / kW;
            int kw = rem % kW;
            
            // Compute the corresponding input coordinates
            int i_val = oh + pH - kh;
            int j_val = ow + pW - kw;
            
            // Only process if coordinates align with stride and are within bounds
            if ((i_val % sH == 0) && (j_val % sW == 0)) {
                int i_in = i_val / sH;
                int j_in = j_val / sW;
                if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                    int input_idx = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                    int weight_idx = ((ic * C_out + oc) * kH + kh) * kW + kw;
                    local_sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        
        // Intra-warp reduction
        local_sum = warpReduceSum(local_sum);
        
        // Shared memory reduction among warps in the block
        __shared__ float shared_sum[32]; // Enough for maximum warps per block
        int lane = threadIdx.x & 31;
        int warpId = threadIdx.x >> 5;
        if (lane == 0) {
            shared_sum[warpId] = local_sum;
        }
        __syncthreads();
        
        // Final reduction by the first warp
        if (warpId == 0) {
            local_sum = (lane < ((blockDim.x + 31) / 32) ? shared_sum[lane] : 0.0f);
            local_sum = warpReduceSum(local_sum);
            if (lane == 0) {
                if (bias != nullptr) {
                    local_sum += bias[oc];
                }
                output[out_idx] = local_sum;
            }
        }
    }
}

// Host function wrapper for the combined kernel
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    
    int kH = weight.size(2);
    int kW = weight.size(3);
    int C_out = weight.size(1);
    
    int sH = stride[0];
    int sW = stride[1];
    int pH = padding[0];
    int pW = padding[1];
    
    // Compute output dimensions for transposed convolution
    int H_out = (H_in - 1) * sH - 2 * pH + kH;
    int W_out = (W_in - 1) * sW - 2 * pW + kW;
    
    torch::Tensor bias;
    const float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
        bias_ptr = bias.data_ptr<float>();
    }
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());
    
    int total_outputs = N * C_out * H_out * W_out;
    // Configure kernel launch (clamp grid size to a reasonable maximum, e.g., 1024 blocks)
    int threads = 256;
    int blocks = total_outputs < 1024 ? total_outputs : 1024;
    
    conv_transpose2d_forward_kernel_combined<<<blocks, threads>>>(
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
    m.def("forward", &conv_transpose2d_forward, "Optimized Conv Transpose 2D forward combining loop unrolling and warp-level reduction",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
