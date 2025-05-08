#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tile dimensions for output block
#define TILE_X 8
#define TILE_Y 8
// Maximum number of floats that can be stored in constant memory
#define MAX_WEIGHT_SIZE 16384

// Declare constant memory for convolution weights
__constant__ float d_weight[MAX_WEIGHT_SIZE];

// CUDA kernel using constant memory for weights
__global__ void conv2d_const_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Compute combined batch and output channel index from blockIdx.z
    int b_oc = blockIdx.z;
    int b = b_oc / out_channels;
    int oc = b_oc % out_channels;

    // Compute output pixel coordinates
    int out_x = blockIdx.x * TILE_X + threadIdx.x;
    int out_y = blockIdx.y * TILE_Y + threadIdx.y;
    if (out_x >= out_width || out_y >= out_height) return;

    float sum = 0.0f;

    // Calculate shared memory tile dimensions
    int tile_w = TILE_X * stride + (kernel_size - 1) * dilation;
    int tile_h = TILE_Y * stride + (kernel_size - 1) * dilation;

    // Compute the top-left corner of the input tile for this block
    int in_tile_origin_x = blockIdx.x * TILE_X * stride - padding;
    int in_tile_origin_y = blockIdx.y * TILE_Y * stride - padding;

    extern __shared__ float shared_tile[]; // Shared memory for input patch

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ic++) {
        // Load the input patch for the current channel into shared memory
        int tile_size = tile_w * tile_h;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = tid; i < tile_size; i += blockDim.x * blockDim.y) {
            int tx = i % tile_w;
            int ty = i / tile_w;
            int in_x = in_tile_origin_x + tx;
            int in_y = in_tile_origin_y + ty;
            int sh_idx = ty * tile_w + tx;
            if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                shared_tile[sh_idx] = input[b * in_channels * in_height * in_width +
                                             ic * in_height * in_width +
                                             in_y * in_width + in_x];
            } else {
                shared_tile[sh_idx] = 0.0f;
            }
        }
        __syncthreads();

        // Compute local coordinates within the shared memory tile for the output pixel
        int local_x = threadIdx.x * stride;
        int local_y = threadIdx.y * stride;

        // Loop over the kernel window and accumulate results using weights from constant memory
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                int sh_x = local_x + kx * dilation;
                int sh_y = local_y + ky * dilation;
                float in_val = shared_tile[sh_y * tile_w + sh_x];
                int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                 ic * kernel_size * kernel_size +
                                 ky * kernel_size + kx;
                float w = d_weight[weight_idx];
                sum += in_val * w;
            }
        }
        __syncthreads(); // Prepare for next input channel
    }

    if (bias) {
        sum += bias[oc];
    }

    output[b * out_channels * out_height * out_width +
           oc * out_height * out_width +
           out_y * out_width + out_x] = sum;
}

// Forward function that uses the constant memory kernel
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

    // Only support non-grouped convolution and weights that fit in constant memory
    TORCH_CHECK(groups == 1, "Only groups==1 is supported in conv2d_const_mem_kernel");
    int weight_numel = weight.numel();
    size_t weight_bytes = weight_numel * sizeof(float);
    TORCH_CHECK(weight_numel <= MAX_WEIGHT_SIZE, "Weight tensor too large for constant memory");

    // Copy weight data into constant memory
    cudaMemcpyToSymbol(d_weight, weight.data_ptr<float>(), weight_bytes);

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width  = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2); // assuming square kernel

    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width  = (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    // Grid dimensions
    dim3 grid((out_width + TILE_X - 1) / TILE_X,
              (out_height + TILE_Y - 1) / TILE_Y,
              batch * out_channels);
    dim3 block(TILE_X, TILE_Y);

    // Shared memory size per block
    int tile_w = TILE_X * stride + (kernel_size - 1) * dilation;
    int tile_h = TILE_Y * stride + (kernel_size - 1) * dilation;
    size_t shared_memory_size = tile_w * tile_h * sizeof(float);

    conv2d_const_kernel<<<grid, block, shared_memory_size>>>(
        x.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_size, stride, padding, dilation);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Convolution forward using constant memory for weights");
}
