#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 16
#define WARPSIZE 32

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B,
    int C_in,
    int C_out,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    int K_H,
    int K_W,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups) {

  extern __shared__ float smem[];
  float* input_tile = smem;
  float* weight_tile = &smem[TILE * TILE];

  int c_out = blockIdx.z;
  int h_out = blockIdx.y * blockDim.y + threadIdx.y;
  int w_out = blockIdx.x * blockDim.x + threadIdx.x;
  
  float sum = 0.0f;

  for (int b = 0; b < B; ++b) {
    for (int c_in = 0; c_in < C_in; ++c_in) {
      for (int kh = 0; kh < K_H; ++kh) {
        for (int kw = 0; kw < K_W; ++kw) {
          int h_in = (h_out - kh * dilation_h + pad_h) / stride_h;
          int w_in = (w_out - kw * dilation_w + pad_w) / stride_w;
          
          if (h_in >= 0 && w_in >= 0 && h_in < H_in && w_in < W_in && 
              (h_out - kh * dilation_h + pad_h) % stride_h == 0 &&
              (w_out - kw * dilation_w + pad_w) % stride_w == 0) {
            
            float val = input[b * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in];
            float w = weight[c_out * C_in * K_H * K_W + c_in * K_H * K_W + kh * K_W + kw];
            sum += val * w;
          }
        }
      }
    }
  }

  // Warp-level reduction
  for (int offset = WARPSIZE/2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  if (threadIdx.x % WARPSIZE == 0 && h_out < H_out && w_out < W_out) {
    if (bias) sum += bias[c_out];
    output[b * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out] = sum;
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
    
  auto input = x.contiguous();
  auto weights = weight.contiguous();
  
  int B = input.size(0);
  int C_in = input.size(1);
  int H_in = input.size(2);
  int W_in = input.size(3);
  
  int C_out = weights.size(1) * groups;
  int K_H = weights.size(2);
  int K_W = weights.size(3);
  
  int H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (K_H - 1) + output_padding[0] + 1;
  int W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (K_W - 1) + output_padding[1] + 1;
  
  auto output = torch::zeros({B, C_out, H_out, W_out}, input.options());
  
  dim3 blocks((W_out + TILE-1)/TILE, (H_out + TILE-1)/TILE, C_out);
  dim3 threads(TILE, TILE);
  
  conv_transpose2d_kernel<<<blocks, threads, (TILE*TILE + C_out*K_H*K_W)*sizeof(float)>>>(
      input.data_ptr<float>(),
      weights.data_ptr<float>(),
      bias ? bias->data_ptr<float>() : nullptr,
      output.data_ptr<float>(),
      B, C_in, C_out,
      H_in, W_in,
      H_out, W_out,
      K_H, K_W,
      stride[0], stride[1],
      padding[0], padding[1],
      dilation[0], dilation[1],
      groups);
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}
