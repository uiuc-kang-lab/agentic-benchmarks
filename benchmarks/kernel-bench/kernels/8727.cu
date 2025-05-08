#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

const int BLOCK_SIZE = 256;
const int ALIGN_BYTES = 16;

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int outD, int outH, int outW,
    int groups, int in_channels_per_group) {

  int total = N * C_in * D_in * H_in * W_in;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    int w = index % W_in;
    int tmp = index / W_in;
    int h = tmp % H_in;
    tmp /= H_in;
    int d = tmp % D_in;
    tmp /= D_in;
    int c_in = tmp % C_in;
    tmp /= C_in;
    int n = tmp;

    int group = c_in / in_channels_per_group;
    float inp = __ldg(&input[index]);

    for (int kd = 0; kd < kernel_d; kd++) {
      int out_d = d * stride_d - pad_d + kd;
      if (out_d < 0 || out_d >= outD) continue;
      for (int kh = 0; kh < kernel_h; kh++) {
        int out_h = h * stride_h - pad_h + kh;
        if (out_h < 0 || out_h >= outH) continue;
        for (int kw = 0; kw < kernel_w; kw++) {
          int out_w = w * stride_w - pad_w + kw;
          if (out_w < 0 || out_w >= outW) continue;

          for (int oc = 0; oc < (C_out / groups); oc++) {
            int weight_index = (((c_in * (C_out / groups) + oc) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
            float weight_val = __ldg(&weight[weight_index]);
            float out_val = inp * weight_val;

            int oc_global = group * (C_out / groups) + oc;
            int output_index = (((n * C_out + oc_global) * outD + out_d) * outH + out_h) * outW + out_w;

            if ((reinterpret_cast<uintptr_t>(&output[output_index]) % ALIGN_BYTES) == 0) {
              atomicAdd(reinterpret_cast<float*>(__builtin_assume_aligned(&output[output_index], ALIGN_BYTES)), out_val);
            } else {
              atomicAdd(&output[output_index], out_val);
            }
          }
        }
      }
    }
  }
}

__global__ void add_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int total, int C_out, int outD, int outH, int outW) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    int c = (index / (outD * outH * outW)) % C_out;
    output[index] += __ldg(&bias[c]);
  }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

  CHECK_INPUT(input);
  CHECK_INPUT(weight);

  int N = input.size(0);
  int C_in = input.size(1);
  int D_in = input.size(2);
  int H_in = input.size(3);
  int W_in = input.size(4);

  int kernel_d = weight.size(2);
  int kernel_h = weight.size(3);
  int kernel_w = weight.size(4);

  int outD = (D_in-1)*stride[0]-2*padding[0]+kernel_d+output_padding[0];
  int outH = (H_in-1)*stride[1]-2*padding[1]+kernel_h+output_padding[1];
  int outW = (W_in-1)*stride[2]-2*padding[2]+kernel_w+output_padding[2];

  int C_out = weight.size(1) * groups;
  auto output = torch::zeros({N, C_out, outD, outH, outW}, input.options());

  int total_input = N * C_in * D_in * H_in * W_in;
  conv_transpose3d_kernel<<<(total_input + BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      output.data_ptr<float>(),
      N, C_in, D_in, H_in, W_in,
      C_out, kernel_d, kernel_h, kernel_w,
      stride[0], stride[1], stride[2],
      padding[0], padding[1], padding[2],
      outD, outH, outW,
      groups, C_in/groups);

  if (bias.has_value()) {
    CHECK_INPUT(*bias);
    int total_output = N * C_out * outD * outH * outW;
    add_bias_kernel<<<(total_output + BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        output.data_ptr<float>(),
        bias->data_ptr<float>(),
        total_output, C_out, outD, outH, outW);
  }

  cudaDeviceSynchronize();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D with LDG/memalign");
}
