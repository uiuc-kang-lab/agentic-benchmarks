#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x);

// Tile dimensions for block configuration
const int TILE_W = 8;
const int TILE_H = 8;
const int TILE_D = 4;

// This kernel assigns one thread per output element. The output tensor has 5 dimensions: [N, C_out, outD, outH, outW].
// We map the spatial dimensions (outW, outH) to blockIdx.x and blockIdx.y and combine the depth along with the batch and channel dimensions into blockIdx.z.
// Each thread computes its (n, oc, od, oh, ow) coordinate and then gathers input contributions by looping over the kernel dimensions and the corresponding input channels.
__global__ void output_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int N,
    const int C_in,
    const int D_in,
    const int H_in,
    const int W_in,
    const int C_out,
    const int outD,
    const int outH,
    const int outW,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int groups,
    const int in_channels_per_group,
    const int C_out_per_group
) {
    // Map blockIdx.x and threadIdx.x to output width
    int ow = blockIdx.x * TILE_W + threadIdx.x;
    // Map blockIdx.y and threadIdx.y to output height
    int oh = blockIdx.y * TILE_H + threadIdx.y;

    // Use blockIdx.z and threadIdx.z to cover the combination of (n, oc, od).
    int combined = blockIdx.z * TILE_D + threadIdx.z;
    int total_combined = N * C_out * outD;
    if (combined >= total_combined) return;
    int od = combined % outD;
    int temp = combined / outD;
    int oc = temp % C_out;
    int n = temp / C_out;

    if (ow >= outW || oh >= outH) return;

    // Determine group for the output channel and its intra-group index
    int group = oc / C_out_per_group;
    int oc_in_group = oc % C_out_per_group;

    float acc = 0.0f;
    
    // Loop over the kernel dimensions. For transposed convolution, we invert the mapping:
    // For each kernel position (kd, kh, kw), compute the candidate input coordinate
    for (int kd = 0; kd < kernel_d; kd++) {
        int id_offset = od + pad_d - kd;
        if (id_offset % stride_d != 0) continue;
        int id = id_offset / stride_d;
        if (id < 0 || id >= D_in) continue;
        for (int kh = 0; kh < kernel_h; kh++) {
            int ih_offset = oh + pad_h - kh;
            if (ih_offset % stride_h != 0) continue;
            int ih = ih_offset / stride_h;
            if (ih < 0 || ih >= H_in) continue;
            for (int kw = 0; kw < kernel_w; kw++) {
                int iw_offset = ow + pad_w - kw;
                if (iw_offset % stride_w != 0) continue;
                int iw = iw_offset / stride_w;
                if (iw < 0 || iw >= W_in) continue;
                
                // Sum over the input channels belonging to the corresponding group
                int start_c = group * in_channels_per_group;
                int end_c = start_c + in_channels_per_group;
                for (int c = start_c; c < end_c; c++) {
                    int input_index = (((n * C_in + c) * D_in + id) * H_in + ih) * W_in + iw;
                    int weight_index = ((((c) * C_out_per_group + oc_in_group) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                    acc += input[input_index] * weight[weight_index];
                }
            }
        }
    }

    int out_index = (((n * C_out + oc) * outD + od) * outH + oh) * outW + ow;
    output[out_index] = acc;
}

// Kernel to add bias to the output tensor. The bias is applied per output channel and broadcasted over (n, outD, outH, outW).
__global__ void add_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int total,
    int C_out,
    int outD,
    int outH,
    int outW
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    // Decode index to get output channel (c). The remaining indices (n, outD, outH, outW) are not needed for bias application.
    int ow = index % outW;
    int tmp = index / outW;
    int oh = tmp % outH;
    tmp /= outH;
    int od = tmp % outD;
    tmp /= outD;
    int c = tmp % C_out;
    output[index] += bias[c];
}

// Host function implementing the forward pass of the 3D transposed convolution using multidimensional grid and block mapping.
// It computes the output dimensions based on input, stride, padding, and output_padding, and then launches the kernel that maps threads
// directly to output elements. This avoids the overhead of atomic operations and uses structured thread indexing to improve performance.

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(*bias);
    }

    // Input dimensions: [N, C_in, D_in, H_in, W_in]
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    // Weight dimensions: [C_in, C_out_per_group, kernel_d, kernel_h, kernel_w]
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
    int pad_d = padding[0], pad_h = padding[1], pad_w = padding[2];
    int out_pad_d = output_padding[0], out_pad_h = output_padding[1], out_pad_w = output_padding[2];

    // Compute output dimensions for transposed convolution
    int outD = (D_in - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
    int outH = (H_in - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
    int outW = (W_in - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;

    int C_out_per_group = weight.size(1);
    int C_out = groups * C_out_per_group;
    int in_channels_per_group = C_in / groups;

    // Allocate the output tensor. Each output element is computed exclusively by one thread, so initialization need not be zeroed.
    auto output = torch::empty({N, C_out, outD, outH, outW}, input.options());

    // Set up a multidimensional grid for output elements.
    // The spatial dimensions (outW, outH) are mapped to grid.x and grid.y, and the combination of (n, oc, outD) is mapped to grid.z.
    dim3 block(TILE_W, TILE_H, TILE_D);
    int grid_x = (outW + TILE_W - 1) / TILE_W;
    int grid_y = (outH + TILE_H - 1) / TILE_H;
    int total_combined = N * C_out * outD;
    int grid_z = (total_combined + TILE_D - 1) / TILE_D;
    dim3 grid(grid_x, grid_y, grid_z);

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    output_conv_transpose3d_kernel<<<grid, block>>>(
        input_ptr, weight_ptr, output_ptr,
        N, C_in, D_in, H_in, W_in,
        C_out, outD, outH, outW,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        groups, in_channels_per_group, C_out_per_group
    );

    // Launch bias addition kernel if bias is provided
    if (bias.has_value()) {
        const float* bias_ptr = (*bias).data_ptr<float>();
        int total_elements = N * C_out * outD * outH * outW;
        int threads = 256;
        int blocks_bias = (total_elements + threads - 1) / threads;
        add_bias_kernel<<<blocks_bias, threads>>>(output_ptr, bias_ptr, total_elements, C_out, outD, outH, outW);
    }

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed Conv3D forward (CUDA) with multidimensional output indexing");
}
