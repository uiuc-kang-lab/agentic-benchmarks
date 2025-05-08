#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

const int BLOCK_SIZE = 256;
const int WARP_SIZE = 32;

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w) {
    
    __shared__ float shared_weight[32][32];  // Cache for weight tiles
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;  // Warp ID
    const int lane = tid % WARP_SIZE; // Lane within warp
    
    // Block-wide indices
    const int batch_idx = blockIdx.z;
    const int out_c = blockIdx.y;
    const int out_h_base = (blockIdx.x / ((W_out + WARP_SIZE - 1) / WARP_SIZE)) * WARP_SIZE;
    const int out_w_base = (blockIdx.x % ((W_out + WARP_SIZE - 1) / WARP_SIZE)) * WARP_SIZE;
    
    // Thread-specific output positions
    const int out_h = out_h_base + wid;
    const int out_w = out_w_base + lane;
    
    // Early exit if outside output bounds
    if (batch_idx >= N || out_c >= C_out || out_h >= H_out || out_w >= W_out) return;
    
    float sum = 0.0f;
    
    // Process input channels in tiles
    for (int cin = 0; cin < C_in; cin++) {
        // Load weight tile into shared memory
        for (int i = tid; i < min(32, kernel_h * kernel_w); i += BLOCK_SIZE) {
            const int kh = i / kernel_w;
            const int kw = i % kernel_w;
            if (kh < kernel_h && kw < kernel_w) {
                shared_weight[wid][lane] = weight[cin * C_out * kernel_h * kernel_w + 
                                               out_c * kernel_h * kernel_w +
                                               kh * kernel_w + kw];
            }
        }
        __syncthreads();
        
        // Compute contribution from this input channel
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                // Calculate input position
                const int h_diff = out_h + pad_h - kh * dilation_h;
                const int w_diff = out_w + pad_w - kw * dilation_w;
                
                // Check if position is aligned with stride
                const bool h_valid = (h_diff % stride_h) == 0;
                const bool w_valid = (w_diff % stride_w) == 0;
                
                if (h_valid && w_valid) {
                    const int in_h = h_diff / stride_h;
                    const int in_w = w_diff / stride_w;
                    
                    if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
                        const float in_val = input[batch_idx * C_in * H_in * W_in +
                                                 cin * H_in * W_in +
                                                 in_h * W_in + in_w];
                        const float weight_val = shared_weight[kh][kw];
                        sum += in_val * weight_val;
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Add bias if present
    if (bias != nullptr) {
        sum += bias[out_c];
    }
    
    // Write output
    if (out_h < H_out && out_w < W_out) {
        output[batch_idx * C_out * H_out * W_out +
               out_c * H_out * W_out +
               out_h * W_out + out_w] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {
    
    const auto N = x.size(0);
    const auto C_in = x.size(1);
    const auto H_in = x.size(2);
    const auto W_in = x.size(3);
    const auto C_out = weight.size(1);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    
    const auto H_out = (H_in - 1) * stride[0] - 2 * padding[0] + 
                      dilation[0] * (kernel_h - 1) + output_padding[0] + 1;
    const auto W_out = (W_in - 1) * stride[1] - 2 * padding[1] + 
                      dilation[1] * (kernel_w - 1) + output_padding[1] + 1;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());
    
    dim3 blocks(
        (H_out + WARP_SIZE - 1) / WARP_SIZE * ((W_out + WARP_SIZE - 1) / WARP_SIZE),
        C_out,
        N
    );
    
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    
    conv_transpose2d_kernel<<<blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_h, kernel_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1]
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}