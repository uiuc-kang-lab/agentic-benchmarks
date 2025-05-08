#include <torch/extension.h>

// Input checking macro
#define CHECK_INPUT(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor"); \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor");

template <int BLOCK_SIZE = 256>
__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weights,
    float* output,
    const int N, const int C, const int D, const int H, const int W,
    const int kD, const int kH, const int kW,
    const int strideD, const int strideH, const int strideW,
    const int padD, const int padH, const int padW,
    const int groups) {
    
    const unsigned FULL_MASK = 0xffffffff;
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    
    // Calculate output position
    const int idx = blockIdx.x * BLOCK_SIZE + tid;
    const int total_elements = N * C * D * H * W;
    
    if (idx >= total_elements) return;
    
    // Decompose index into position
    int n = idx / (C * D * H * W);
    int c = (idx / (D * H * W)) % C;
    int d = (idx / (H * W)) % D;
    int h = (idx / W) % H;
    int w = idx % W;
    
    float sum = 0.0f;
    const int group_size = C / groups;
    const int g = c / group_size;
    
    // Compute convolution using warp-level primitives
    #pragma unroll
    for (int kd = 0; kd < kD; kd++) {
        #pragma unroll
        for (int kh = 0; kh < kH; kh++) {
            #pragma unroll
            for (int kw = 0; kw < kW; kw++) {
                int id = d * strideD - padD + kd;
                int ih = h * strideH - padH + kh;
                int iw = w * strideW - padW + kw;
                
                if (id >= 0 && id < D && ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float val = input[n * C * D * H * W + c * D * H * W + id * H * W + ih * W + iw];
                    float weight = weights[g * group_size * kD * kH * kW + c % group_size * kD * kH * kW + kd * kH * kW + kh * kW + kw];
                    sum += val * weight;
                }
            }
        }
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    
    // First thread in each warp writes the result
    if (lane == 0) {
        output[idx] = sum;
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
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(*bias);
    }
    
    auto output = torch::zeros_like(x);
    
    const int N = x.size(0);
    const int C = x.size(1);
    const int D = x.size(2);
    const int H = x.size(3);
    const int W = x.size(4);
    
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);
    
    const dim3 blocks((N * C * D * H * W + 255) / 256);
    const dim3 threads(256);
    
    conv_transpose3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        kD, kH, kW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        groups
    );
    
    if (bias.has_value()) {
        output.add_(*bias);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed Conv3D forward (CUDA)");
}