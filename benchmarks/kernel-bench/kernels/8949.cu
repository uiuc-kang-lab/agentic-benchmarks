#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

__global__ void initialize_output_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch, int out_channels,
    int out_h, int out_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch * out_channels * out_h * out_w) return;
  
  int ow = idx % out_w;
  int oh = (idx / out_w) % out_h;
  int oc = (idx / (out_w * out_h)) % out_channels;
  output[idx] = bias[oc];
}

__global__ void conv_transposed_shared_weights_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch, int in_channels, int in_h, int in_w,
    int out_channels_per_group, int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups, int out_h, int out_w,
    int in_channels_per_group) {

  extern __shared__ float s_weights[];
  
  int g = blockIdx.y;
  int n = blockIdx.z;
  int tid = threadIdx.x;
  
  // Load group weights into shared memory
  int weight_size = in_channels_per_group * out_channels_per_group * kernel_h * kernel_w;
  for(int i = tid; i < weight_size; i += blockDim.x) {
    s_weights[i] = weight[g * weight_size + i];
  }
  __syncthreads();

  // Process input elements for this group and batch
  for(int c_in_group = 0; c_in_group < in_channels_per_group; c_in_group++) {
    for(int ih = tid / in_w; ih < in_h; ih += blockDim.x / in_w) {
      int iw = tid % in_w;
      if(iw >= in_w) continue;
      
      float x_val = x[n * in_channels * in_h * in_w +
                     (g * in_channels_per_group + c_in_group) * in_h * in_w +
                     ih * in_w + iw];

      for(int kh = 0; kh < kernel_h; kh++) {
        int oh = ih * stride_h - pad_h + kh * dilation_h;
        if(oh < 0 || oh >= out_h) continue;
        
        for(int kw = 0; kw < kernel_w; kw++) {
          int ow = iw * stride_w - pad_w + kw * dilation_w;
          if(ow < 0 || ow >= out_w) continue;
          
          for(int oc = 0; oc < out_channels_per_group; oc++) {
            int weight_idx = c_in_group * (out_channels_per_group * kernel_h * kernel_w)
                           + oc * (kernel_h * kernel_w)
                           + kh * kernel_w + kw;
            
            float contrib = x_val * s_weights[weight_idx];
            int out_idx = n * (groups * out_channels_per_group * out_h * out_w)
                        + (g * out_channels_per_group + oc) * (out_h * out_w)
                        + oh * out_w + ow;
            
            atomicAdd(&output[out_idx], contrib);
          }
        }
      }
    }
  }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {
  
  x = x.contiguous();
  weight = weight.contiguous();
  
  int batch = x.size(0);
  int in_channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);

  int kernel_h = weight.size(2);
  int kernel_w = weight.size(3);
  int out_channels_per_group = weight.size(1);
  int out_channels = out_channels_per_group * groups;
  
  int stride_h = stride[0], stride_w = stride[1];
  int pad_h = padding[0], pad_w = padding[1];
  int dilation_h = dilation[0], dilation_w = dilation[1];

  int out_h = (in_h-1)*stride_h - 2*pad_h + dilation_h*(kernel_h-1) + 1;
  int out_w = (in_w-1)*stride_w - 2*pad_w + dilation_w*(kernel_w-1) + 1;

  auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());
  
  if (!bias.has_value() || !bias.value().defined())
    bias = at::zeros({out_channels}, weight.options());
  else
    bias = bias.value().contiguous();

  // Initialize output
  int total_out = batch * out_channels * out_h * out_w;
  initialize_output_kernel<<<(total_out+511)/512,512>>>(
      output.data_ptr<float>(),
      bias.value().data_ptr<float>(),
      batch, out_channels, out_h, out_w);

  // Main kernel
  int in_channels_per_group = in_channels / groups;
  dim3 blocks(1, groups, batch);
  int threads = 256;
  int shared_mem = in_channels_per_group * out_channels_per_group * kernel_h * kernel_w * sizeof(float);
  
  conv_transposed_shared_weights_kernel<<<blocks, threads, shared_mem>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      output.data_ptr<float>(),
      batch, in_channels, in_h, in_w,
      out_channels_per_group, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w,
      dilation_h, dilation_w, groups,
      out_h, out_w, in_channels_per_group);

  cudaDeviceSynchronize();

// Launch a separate kernel for output initialization if needed
if (total_out > 1000000) {
    int init_blocks = (total_out + 511) / 512;
    initialize_output_kernel<<<init_blocks, 512>>>(
        output.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        batch, out_channels, out_h, out_w);
} else {
    initialize_output_kernel<<<(total_out + 511) / 512, 512>>>(
        output.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        batch, out_channels, out_h, out_w);
}
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "2D Transposed Conv with Shared Weights (CUDA)");
}