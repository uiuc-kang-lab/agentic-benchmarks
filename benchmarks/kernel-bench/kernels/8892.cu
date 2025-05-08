#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

#define BLOCK_SIZE_X 8  // output channels
#define BLOCK_SIZE_Y 8  // height
#define BLOCK_SIZE_Z 8  // width

__device__ __forceinline__ int gcd(int a, int b) {
    while(b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

__global__ void conv_transpose2d_kernel_3d(
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

    // Calculate output position using 3D block and thread organization
    const int oc = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int oh = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int ow = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z;
    
    // Early exit if outside output dimensions
    if (oc >= out_channels || oh >= out_h || ow >= out_w)
        return;

    // Process each batch in sequence
    for (int n = 0; n < batch; n++) {
        float out_val = bias[oc];
        const int g = oc / out_channels_per_group;

        // Calculate padding offsets
        const int candidate_h = oh + pad_h;
        const int candidate_w = ow + pad_w;

        // Calculate stride and dilation parameters for height
        const int mod_h = candidate_h % stride_h;
        int offset_kh = -1;
        for (int k = 0; k < stride_h; k++) {
            if ((k * dilation_h) % stride_h == mod_h) {
                offset_kh = k;
                break;
            }
        }
        const int step_kh = stride_h / gcd(stride_h, dilation_h);

        // Calculate stride and dilation parameters for width
        const int mod_w = candidate_w % stride_w;
        int offset_kw = -1;
        for (int k = 0; k < stride_w; k++) {
            if ((k * dilation_w) % stride_w == mod_w) {
                offset_kw = k;
                break;
            }
        }
        const int step_kw = stride_w / gcd(stride_w, dilation_w);

        // Process valid kernel positions
        #pragma unroll 4
        for (int kh = offset_kh; kh < kernel_h; kh += step_kh) {
            const int h_in_candidate = candidate_h - kh * dilation_h;
            if (h_in_candidate < 0 || (h_in_candidate % stride_h) != 0) continue;
            const int ih = h_in_candidate / stride_h;
            if (ih >= in_h) continue;

            #pragma unroll 4
            for (int kw = offset_kw; kw < kernel_w; kw += step_kw) {
                const int w_in_candidate = candidate_w - kw * dilation_w;
                if (w_in_candidate < 0 || (w_in_candidate % stride_w) != 0) continue;
                const int iw = w_in_candidate / stride_w;
                if (iw >= in_w) continue;

                // Process input channels for current group
                #pragma unroll 4
                for (int c = g * in_channels_per_group; c < (g + 1) * in_channels_per_group; c++) {
                    const int x_idx = ((n * in_channels + c) * in_h + ih) * in_w + iw;
                    const int w_idx = ((c * out_channels_per_group + 
                                    (oc - g * out_channels_per_group)) * kernel_h + kh) * kernel_w + kw;
                    
                    out_val += __ldg(&x[x_idx]) * __ldg(&weight[w_idx]);
                }
            }
        }

        // Write output
        const int out_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
        output[out_idx] = out_val;
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

    // Calculate grid dimensions for 3D block organization
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 blocks(
        (out_channels + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (out_h + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        (out_w + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z
    );

    conv_transpose2d_kernel_3d<<<blocks, threads>>>(
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
        in_channels / groups,
        out_channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Block Optimized 2D Transposed Convolution (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}