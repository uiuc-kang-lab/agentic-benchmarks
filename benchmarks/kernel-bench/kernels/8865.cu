#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// Define shared memory size for weights
#define BLOCK_SIZE 32
#define WARP_SIZE 32
#define SHARED_MEM_SIZE 1024

__global__ void conv_transpose2d_coalesced_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int in_channels_per_group,
    const int out_channels_per_group) {

    __shared__ float shared_weight[SHARED_MEM_SIZE];
    
    // Calculate output position ensuring coalesced memory access
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Block processes consecutive output elements for better coalescing
    const int block_offset = blockIdx.x * blockDim.x;
    const int global_tid = block_offset + tid;
    
    // Calculate output indices ensuring consecutive threads access consecutive memory
    const int ow = global_tid % out_w;
    int tmp = global_tid / out_w;
    const int oh = tmp % out_h;
    tmp = tmp / out_h;
    const int oc = tmp % out_channels;
    const int n = tmp / out_channels;

    if (global_tid >= batch * out_channels * out_h * out_w) return;

    // Calculate group
    const int g = oc / out_channels_per_group;
    float sum = bias[oc];

    // Pre-calculate input bounds
    const int h_in_start = oh + pad_h;
    const int w_in_start = ow + pad_w;

    // Process input channels in chunks that fit in shared memory
    for (int c_start = g * in_channels_per_group; c_start < (g + 1) * in_channels_per_group; c_start += SHARED_MEM_SIZE / (kernel_h * kernel_w)) {
        const int c_end = min(c_start + SHARED_MEM_SIZE / (kernel_h * kernel_w), (g + 1) * in_channels_per_group);
        
        // Cooperatively load weights into shared memory
        for (int i = tid; i < (c_end - c_start) * kernel_h * kernel_w && i < SHARED_MEM_SIZE; i += blockDim.x) {
            const int c_offset = c_start + (i / (kernel_h * kernel_w));
            const int kh = (i % (kernel_h * kernel_w)) / kernel_w;
            const int kw = i % kernel_w;
            shared_weight[i] = weight[c_offset * (out_channels_per_group * kernel_h * kernel_w) +
                                    (oc - g * out_channels_per_group) * (kernel_h * kernel_w) +
                                    kh * kernel_w + kw];
        }
        __syncthreads();

        // Process loaded channels
        for (int c = c_start; c < c_end; c++) {
            const int weight_base = (c - c_start) * kernel_h * kernel_w;
            
            #pragma unroll
            for (int kh = 0; kh < kernel_h; kh++) {
                const int h_in = h_in_start - kh * dilation_h;
                if (h_in % stride_h != 0) continue;
                const int ih = h_in / stride_h;
                if (ih < 0 || ih >= in_h) continue;

                #pragma unroll
                for (int kw = 0; kw < kernel_w; kw++) {
                    const int w_in = w_in_start - kw * dilation_w;
                    if (w_in % stride_w != 0) continue;
                    const int iw = w_in / stride_w;
                    if (iw < 0 || iw >= in_w) continue;

                    const int x_idx = ((n * in_channels + c) * in_h + ih) * in_w + iw;
                    const int weight_idx = weight_base + kh * kernel_w + kw;
                    
                    sum += x[x_idx] * shared_weight[weight_idx];
                }
            }
        }
        __syncthreads();
    }

    // Write output with coalesced access pattern
    const int out_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
    output[out_idx] = sum;
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
    if (bias.has_value() && bias.value().defined())
        bias = bias.value().contiguous();
    
    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels_per_group = weight.size(1);
    const int out_channels = out_channels_per_group * groups;
    
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];
    
    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;
    
    if (!bias.has_value() || !bias.value().defined()) {
        bias = at::zeros({out_channels}, weight.options());
    }
    
    auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());
    
    int in_channels_per_group = in_channels / groups;
    
    // Configure kernel launch parameters for coalesced access
    const int threads = 256;
    const int total_elements = batch * out_channels * out_h * out_w;
    const int blocks = (total_elements + threads - 1) / threads;
    
    conv_transpose2d_coalesced_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        in_channels_per_group,
        out_channels_per_group
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Transposed Convolution with Coalesced Memory Access (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}