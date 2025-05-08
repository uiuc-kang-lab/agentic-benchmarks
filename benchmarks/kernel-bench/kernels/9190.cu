#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// Optimized CUDA kernel for conv_transpose2d using improved thread/block indexing
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    const int N,
    const int C_in,
    const int H_in,
    const int W_in,
    const int C_out,
    const int H_out,
    const int W_out,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w) {

    // Compute spatial indices from block and thread indices
    int ow = blockIdx.x * blockDim.x + threadIdx.x;  // Output width index
    int oh = blockIdx.y * blockDim.y + threadIdx.y;  // Output height index

    // Allocate shared memory for input and weight tiles
    extern __shared__ float shared_mem[];
    float* shared_x = shared_mem;
    float* shared_weight = shared_mem + blockDim.x * blockDim.y;

    // Load input and weight tiles into shared memory
    int x_tile_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int weight_tile_idx = threadIdx.y * blockDim.x + threadIdx.x;
    if (x_tile_idx < C_in * H_in * W_in) {
        shared_x[x_tile_idx] = x[x_tile_idx];
    }
    if (weight_tile_idx < C_in * C_out * kernel_h * kernel_w) {
        shared_weight[weight_tile_idx] = weight[weight_tile_idx];
    }
    __syncthreads();

    // Combine batch and output channel into one grid dimension
    int combined = blockIdx.z; 
    if (combined >= N * C_out) return;  
    int n = combined / C_out;
    int oc = combined % C_out;

    if (oh < H_out && ow < W_out) {
        // Initialize output with bias if provided
        float sum = (bias != nullptr) ? bias[oc] : 0.0f;
        
        // Loop over input channels and the kernel spatial dimensions
        for (int ic = 0; ic < C_in; ic++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    // Compute corresponding input coordinates using transposed convolution formula
                    int i_in = oh + pad_h - kh;
                    int j_in = ow + pad_w - kw;

                    // Only add contributions if coordinates align with the stride
                    if ((i_in % stride_h == 0) && (j_in % stride_w == 0)) {
                        int ih = i_in / stride_h;
                        int iw = j_in / stride_w;

                        if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                            int x_idx = n * (C_in * H_in * W_in) + ic * (H_in * W_in) + ih * W_in + iw;
                            int w_idx = ic * (C_out * kernel_h * kernel_w) + oc * (kernel_h * kernel_w) + kh * kernel_w + kw;
                            sum += x[x_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        
        int y_idx = n * (C_out * H_out * W_out) + oc * (H_out * W_out) + oh * W_out + ow;
        y[y_idx] = sum;
    }
}

// Host function that sets up and launches the CUDA kernel
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    // Extract input tensor dimensions
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);

    // Extract kernel dimensions (assumes weight layout: [C_in, C_out, kernel_h, kernel_w])
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int C_out = weight.size(1);

    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];

    // Compute output spatial dimensions for transposed convolution
    const int H_out = (H_in - 1) * stride_h - 2 * pad_h + kernel_h;
    const int W_out = (W_in - 1) * stride_w - 2 * pad_w + kernel_w;

    // Allocate output tensor on the same device as input
    auto y = torch::zeros({N, C_out, H_out, W_out}, x.options());

    // Handle the optional bias pointer
    const float* bias_ptr = nullptr;
    torch::Tensor bias_tensor;
    if (!bias_obj.is_none()) {
        bias_tensor = bias_obj.cast<torch::Tensor>();
        bias_ptr = bias_tensor.data_ptr<float>();
    }

    // Define CUDA kernel launch configuration; using 16x16 blocks for spatial dimensions
    dim3 block(16, 16);
    dim3 grid((W_out + block.x - 1) / block.x,
              (H_out + block.y - 1) / block.y,
              N * C_out);

    // Launch the kernel
    conv_transpose2d_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w
    );
    cudaDeviceSynchronize();

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Optimized Conv Transpose 2D forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
