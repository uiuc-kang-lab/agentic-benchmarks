#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int N, const int C, const int D, const int H, const int W,
    const int K, const int T, const int R, const int S,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int out_d, const int out_h, const int out_w,
    const int groups) {

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int batch_idx = blockIdx.x;
    const int out_ch = blockIdx.y;
    const int out_z = blockIdx.z / (out_h * out_w);
    const int out_y = (blockIdx.z % (out_h * out_w)) / out_w;
    const int out_x = blockIdx.z % out_w;
    
    const int group_size = C / groups;
    const int group_id = out_ch / (K / groups);
    
    float sum = 0.0f;
    
    if (batch_idx < N && out_ch < K && out_z < out_d && out_y < out_h && out_x < out_w) {
        for (int c = lane_id; c < group_size; c += WARP_SIZE) {
            const int in_c = group_id * group_size + c;
            
            for (int kt = 0; kt < T; kt++) {
                for (int kr = 0; kr < R; kr++) {
                    for (int ks = 0; ks < S; ks++) {
                        int in_z = out_z + pad_d - kt;
                        int in_y = out_y + pad_h - kr;
                        int in_x = out_x + pad_w - ks;
                        
                        if (in_z % stride_d == 0 && in_y % stride_h == 0 && in_x % stride_w == 0) {
                            in_z /= stride_d;
                            in_y /= stride_h;
                            in_x /= stride_w;
                            
                            if (in_z >= 0 && in_z < D && in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                const float in_val = input[((batch_idx * C + in_c) * D + in_z) * H * W +
                                                          in_y * W + in_x];
                                const float w_val = weight[((out_ch * group_size + c) * T + kt) * R * S +
                                                          kr * S + ks];
                                sum += in_val * w_val;
                            }
                        }
                    }
                }
            }
        }
        
        // Warp reduction using shuffle
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        }
        
        if (lane_id == 0) {
            float final_sum = sum;
            if (bias != nullptr) {
                final_sum += bias[out_ch];
            }
            output[((batch_idx * K + out_ch) * out_d + out_z) * out_h * out_w +
                   out_y * out_w + out_x] = final_sum;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {
    
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    
    auto N = x.size(0);
    auto C = x.size(1);
    auto D = x.size(2);
    auto H = x.size(3);
    auto W = x.size(4);
    
    auto K = weight.size(1) * groups;
    auto T = weight.size(2);
    auto R = weight.size(3);
    auto S = weight.size(4);
    
    auto stride_d = stride[0];
    auto stride_h = stride[1];
    auto stride_w = stride[2];
    
    auto pad_d = padding[0];
    auto pad_h = padding[1];
    auto pad_w = padding[2];
    
    auto out_d = (D - 1) * stride_d - 2 * pad_d + T + output_padding[0];
    auto out_h = (H - 1) * stride_h - 2 * pad_h + R + output_padding[1];
    auto out_w = (W - 1) * stride_w - 2 * pad_w + S + output_padding[2];
    
    auto output = torch::zeros({N, K, out_d, out_h, out_w}, x.options());
    
    const dim3 threads(WARP_SIZE);
    const dim3 blocks(N, K, out_d * out_h * out_w);
    
    conv_transpose3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, D, H, W,
        K, T, R, S,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_d, out_h, out_w,
        groups
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed Conv3D forward (CUDA)");
}