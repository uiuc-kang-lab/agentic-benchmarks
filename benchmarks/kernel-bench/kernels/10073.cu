#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE + 2) // Include padding for the convolution window
#define MAX_KERNEL_SIZE 7

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k, int stride, int padding, int dilation) {
    
    __shared__ scalar_t shared_input[TILE_SIZE][TILE_SIZE];
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;
    
    // Calculate base indices
    const int block_x = blockIdx.x * BLOCK_SIZE;
    const int block_y = blockIdx.y * BLOCK_SIZE;
    
    // Handle multiple elements per thread using stride loop
    for (int work_item = blockIdx.z; work_item < batch * channels; work_item += gridDim.z) {
        const int n = work_item / channels;
        const int c = work_item % channels;
        
        // Load input tile into shared memory
        for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += total_threads) {
            const int tile_row = i / TILE_SIZE;
            const int tile_col = i % TILE_SIZE;
            const int in_row = block_y + tile_row - padding;
            const int in_col = block_x + tile_col - padding;
            
            if (in_row >= 0 && in_row < in_h && in_col >= 0 && in_col < in_w) {
                shared_input[tile_row][tile_col] = input[
                    n * channels * in_h * in_w +
                    c * in_h * in_w +
                    in_row * in_w +
                    in_col];
            } else {
                shared_input[tile_row][tile_col] = 0;
            }
        }
        __syncthreads();
        
        // Process output elements
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        
        for (int oh = ty; oh < min(BLOCK_SIZE, out_h - block_y); oh += blockDim.y) {
            for (int ow = tx; ow < min(BLOCK_SIZE, out_w - block_x); ow += blockDim.x) {
                scalar_t sum = 0;
                
                #pragma unroll
                for (int i = 0; i < k; i++) {
                    #pragma unroll
                    for (int j = 0; j < k; j++) {
                        const int sh_row = oh * stride + i;
                        const int sh_col = ow * stride + j;
                        sum += shared_input[sh_row][sh_col] * 
                               weight[c * k * k + i * k + j];
                    }
                }
                
                if (bias != nullptr) {
                    sum += bias[c];
                }
                
                const int out_idx = n * channels * out_h * out_w +
                                  c * out_h * out_w +
                                  (block_y + oh) * out_w +
                                  (block_x + ow);
                output[out_idx] = sum;
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch, int in_channels, int out_channels,
    int h, int w) {
    
    __shared__ scalar_t shared_weight[BLOCK_SIZE][BLOCK_SIZE];
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;
    
    // Calculate base indices
    const int block_x = blockIdx.x * BLOCK_SIZE;
    const int block_y = blockIdx.y * BLOCK_SIZE;
    
    // Handle multiple elements per thread using stride loop
    for (int work_item = blockIdx.z; work_item < batch * out_channels; work_item += gridDim.z) {
        const int n = work_item / out_channels;
        const int oc = work_item % out_channels;
        
        scalar_t sum = 0;
        
        // Process input channels in tiles
        for (int ic_block = 0; ic_block < in_channels; ic_block += BLOCK_SIZE) {
            // Load weight tile into shared memory
            for (int i = tid; i < BLOCK_SIZE * BLOCK_SIZE; i += total_threads) {
                const int row = i / BLOCK_SIZE;
                const int col = i % BLOCK_SIZE;
                if (ic_block + col < in_channels && oc + row < out_channels) {
                    shared_weight[row][col] = weight[(oc + row) * in_channels + ic_block + col];
                } else {
                    shared_weight[row][col] = 0;
                }
            }
            __syncthreads();
            
            // Process spatial elements
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            
            for (int oh = ty; oh < min(BLOCK_SIZE, h - block_y); oh += blockDim.y) {
                for (int ow = tx; ow < min(BLOCK_SIZE, w - block_x); ow += blockDim.x) {
                    scalar_t local_sum = 0;
                    
                    #pragma unroll
                    for (int i = 0; i < BLOCK_SIZE; ++i) {
                        if (ic_block + i < in_channels) {
                            const int in_idx = n * in_channels * h * w +
                                             (ic_block + i) * h * w +
                                             (block_y + oh) * w +
                                             (block_x + ow);
                            local_sum += input[in_idx] * shared_weight[threadIdx.y][i];
                        }
                    }
                    sum += local_sum;
                }
            }
            __syncthreads();
        }
        
        // Write output
        if (threadIdx.x < min(BLOCK_SIZE, w - block_x) &&
            threadIdx.y < min(BLOCK_SIZE, h - block_y)) {
            
            if (bias != nullptr) {
                sum += bias[oc];
            }
            
            const int out_idx = n * out_channels * h * w +
                               oc * h * w +
                               (block_y + threadIdx.y) * w +
                               (block_x + threadIdx.x);
            output[out_idx] = sum;
        }
    }
}

torch::Tensor forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride,
    int padding,
    int dilation) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    
    int k = depthwise_weight.size(2);
    int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    
    auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((out_w + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE,
              min(batch * in_channels, 65535));
    
    const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0) ?
                                    depthwise_bias.data_ptr() : nullptr;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<grid, block>>>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            reinterpret_cast<const scalar_t*>(depthwise_bias_ptr),
            depthwise_output.data_ptr<scalar_t>(),
            batch, in_channels,
            in_h, in_w,
            out_h, out_w,
            k, stride, padding, dilation);
    }));
    
    int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
    
    dim3 grid_pw((out_w + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 min(batch * out_channels, 65535));
    
    const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0) ?
                                    pointwise_bias.data_ptr() : nullptr;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
        pointwise_conv2d_kernel<scalar_t><<<grid_pw, block>>>(
            depthwise_output.data_ptr<scalar_t>(),
            pointwise_weight.data_ptr<scalar_t>(),
            reinterpret_cast<const scalar_t*>(pointwise_bias_ptr),
            output.data_ptr<scalar_t>(),
            batch, in_channels, out_channels,
            out_h, out_w);
    }));
    
    return output;
}

at::Tensor toTensor(const py::object& obj) {
    if (obj.is_none()) {
        return at::Tensor();
    }
    try {
        return obj.cast<at::Tensor>();
    } catch (const py::cast_error& e) {
        if (py::hasattr(obj, "data")) {
            return obj.attr("data").cast<at::Tensor>();
        }
        throw std::runtime_error("Expected a torch Tensor or Parameter.");
    }
}

at::Tensor forward_wrapper(
    py::object x_obj,
    py::object depthwise_weight_obj,
    py::object pointwise_weight_obj,
    py::object depthwise_bias_obj,
    py::object pointwise_bias_obj,
    int stride,
    int padding,
    int dilation) {
    
    auto x = toTensor(x_obj);
    auto depthwise_weight = toTensor(depthwise_weight_obj);
    auto pointwise_weight = toTensor(pointwise_weight_obj);
    auto depthwise_bias = toTensor(depthwise_bias_obj);
    auto pointwise_bias = toTensor(pointwise_bias_obj);
    
    return forward_cuda(x, depthwise_weight, pointwise_weight,
                       depthwise_bias, pointwise_bias,
                       stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward");
}