#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define the tile size each thread will compute
#define THREAD_TILE_R 2
#define THREAD_TILE_C 2

// Evenly distribute workload: each thread computes a tile (THREAD_TILE_R x THREAD_TILE_C) of output pixels
__global__ void even_dist_conv2d_kernel(
    const float * __restrict__ input,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    float * __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_size,
    int out_h,
    int out_w,
    int stride,
    int padding) {

    // Determine output channel and batch index
    int oc = blockIdx.z % out_channels;
    int n  = blockIdx.z / out_channels;

    // Load filter weights for this output channel into shared memory
    extern __shared__ float sh_weight[]; // Size: in_channels * kernel_size * kernel_size
    int filter_elems = in_channels * kernel_size * kernel_size;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int idx = tid; idx < filter_elems; idx += blockDim.x * blockDim.y) {
        int ic = idx / (kernel_size * kernel_size);
        int rem = idx % (kernel_size * kernel_size);
        int ki = rem / kernel_size;
        int kj = rem % kernel_size;
        int weight_index = oc * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + ki * kernel_size + kj;
        sh_weight[idx] = weight[weight_index];
    }
    __syncthreads();

    // Calculate the base output coordinate for the tile computed by this thread
    int thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x;  // tile index in x
    int thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y;  // tile index in y

    int base_out_col = thread_idx_x * THREAD_TILE_C;
    int base_out_row = thread_idx_y * THREAD_TILE_R;

    // Each thread computes a tile of size THREAD_TILE_R x THREAD_TILE_C
    for (int r = 0; r < THREAD_TILE_R; r++) {
        int out_row = base_out_row + r;
        if (out_row >= out_h) continue;
        for (int c = 0; c < THREAD_TILE_C; c++) {
            int out_col = base_out_col + c;
            if (out_col >= out_w) continue;
            float sum = 0.0f;

            // Compute input base for this output pixel
            int in_row_base = out_row * stride - padding;
            int in_col_base = out_col * stride - padding;

            // Check if the entire convolution window is within bounds
            bool fast = (in_row_base >= 0) && (in_col_base >= 0) && 
                        ((in_row_base + kernel_size) <= in_h) && 
                        ((in_col_base + kernel_size) <= in_w);

            if (fast) {
                // Fast path: No bounds checking required, using pointer arithmetic for improved performance
                for (int ic = 0; ic < in_channels; ic++) {
                    const float* input_ptr = input + n * (in_channels * in_h * in_w) +
                                              ic * (in_h * in_w) + in_row_base * in_w + in_col_base;
                    const float* filter_ptr = sh_weight + ic * (kernel_size * kernel_size);
                    for (int ki = 0; ki < kernel_size; ki++) {
                        for (int kj = 0; kj < kernel_size; kj++) {
                            sum += input_ptr[ki * in_w + kj] * filter_ptr[ki * kernel_size + kj];
                        }
                    }
                }
            } else {
                // Slow path: Perform bounds checking
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int ki = 0; ki < kernel_size; ki++) {
                        for (int kj = 0; kj < kernel_size; kj++) {
                            int in_row = out_row * stride - padding + ki;
                            int in_col = out_col * stride - padding + kj;
                            if (in_row >= 0 && in_row < in_h && in_col >= 0 && in_col < in_w) {
                                int input_index = n * (in_channels * in_h * in_w) + ic * (in_h * in_w) + in_row * in_w + in_col;
                                int filter_index = ic * (kernel_size * kernel_size) + ki * kernel_size + kj;
                                sum += input[input_index] * sh_weight[filter_index];
                            }
                        }
                    }
                }
            }
            if (bias != nullptr) {
                sum += bias[oc];
            }
            int out_index = n * (out_channels * out_h * out_w) + oc * (out_h * out_w) + out_row * out_w + out_col;
            output[out_index] = sum;
        }
    }
}


// Host function for forward pass
// Falls back to torch::conv2d for unsupported configurations.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    if (groups != 1 || dilation != 1) {
        if (bias.has_value()) {
            return torch::conv2d(x, weight, bias.value(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        } else {
            return torch::conv2d(x, weight, torch::Tensor(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        }
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // square kernel assumed

    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    // Determine block dimensions. Each thread computes a tile of size THREAD_TILE_R x THREAD_TILE_C.
    // Hence, grid dimensions are adjusted accordingly.
    dim3 block(16, 8, 1);
    int grid_x = (out_w + (block.x * THREAD_TILE_C) - 1) / (block.x * THREAD_TILE_C);
    int grid_y = (out_h + (block.y * THREAD_TILE_R) - 1) / (block.y * THREAD_TILE_R);
    dim3 grid(grid_x, grid_y, batch_size * out_channels);

    size_t shared_mem_size = in_channels * kernel_size * kernel_size * sizeof(float);

    even_dist_conv2d_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_size,
        out_h,
        out_w,
        stride,
        padding
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Evenly distributed CUDA forward function for 2D convolution");
}
