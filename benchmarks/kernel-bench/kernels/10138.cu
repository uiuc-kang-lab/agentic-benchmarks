#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define THREADS_PER_BLOCK 256
#define TILE_SIZE 16

__global__ void depthwise_conv2d_kernel_stride(const float* __restrict__ input,
                                             const float* __restrict__ weight,
                                             const float* __restrict__ bias,
                                             float* __restrict__ output,
                                             int batch,
                                             int channels,
                                             int in_h, int in_w,
                                             int out_h, int out_w,
                                             int k,
                                             int stride,
                                             int padding,
                                             int dilation) {
    __shared__ float shared_weight[9];  // Assuming max kernel size of 3x3
    
    int tid = threadIdx.x;
    int stride_index = blockIdx.x * blockDim.x + tid;
    
    // Load kernel weights into shared memory
    if (tid < k * k) {
        shared_weight[tid] = weight[blockIdx.y * k * k + tid];
    }
    __syncthreads();
    
    while (stride_index < batch * channels * out_h * out_w) {
        int ow = stride_index % out_w;
        int tmp = stride_index / out_w;
        int oh = tmp % out_h;
        tmp = tmp / out_h;
        int c = tmp % channels;
        int n = tmp / channels;
        
        float sum = 0.0f;
        
        #pragma unroll
        for (int i = 0; i < k; ++i) {
            #pragma unroll
            for (int j = 0; j < k; ++j) {
                int ih = oh * stride - padding + i * dilation;
                int iw = ow * stride - padding + j * dilation;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int input_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                    sum += input[input_idx] * shared_weight[i * k + j];
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        output[stride_index] = sum;
        stride_index += gridDim.x * blockDim.x;
    }
}

__global__ void pointwise_conv2d_kernel_stride(const float* __restrict__ input,
                                             const float* __restrict__ weight,
                                             const float* __restrict__ bias,
                                             float* __restrict__ output,
                                             int batch,
                                             int in_channels,
                                             int out_channels,
                                             int h,
                                             int w) {
    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_weight[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int stride_index = (by + ty) * w + (bx + tx);
    int total_outputs = batch * out_channels * h * w;
    
    while (stride_index < total_outputs) {
        float sum = 0.0f;
        
        int ow = stride_index % w;
        int tmp = stride_index / w;
        int oh = tmp % h;
        tmp = tmp / h;
        int oc = tmp % out_channels;
        int n = tmp / out_channels;
        
        for (int tile = 0; tile < (in_channels + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
            // Load input tile
            if ((by + ty) < h && (tile * TILE_SIZE + tx) < in_channels) {
                shared_input[ty][tx] = input[
                    n * in_channels * h * w +
                    (tile * TILE_SIZE + tx) * h * w +
                    (by + ty) * w + bx];
            }
            
            // Load weight tile
            if ((tile * TILE_SIZE + ty) < in_channels && (bx + tx) < out_channels) {
                shared_weight[ty][tx] = weight[
                    oc * in_channels +
                    tile * TILE_SIZE + ty];
            }
            
            __syncthreads();
            
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                if ((tile * TILE_SIZE + k) < in_channels) {
                    sum += shared_input[ty][k] * shared_weight[k][tx];
                }
            }
            
            __syncthreads();
        }
        
        if (stride_index < total_outputs) {
            if (bias != nullptr) {
                sum += bias[oc];
            }
            output[stride_index] = sum;
        }
        
        stride_index += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    }
}

torch::Tensor forward_cuda(const torch::Tensor& x,
                         const torch::Tensor& depthwise_weight,
                         const torch::Tensor& pointwise_weight,
                         const torch::Tensor& depthwise_bias,
                         const torch::Tensor& pointwise_bias,
                         int stride,
                         int padding,
                         int dilation) {
    
    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int k = depthwise_weight.size(2);
    int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    
    auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());
    
    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks((batch * in_channels * out_h * out_w + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, in_channels);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel_stride<<<blocks, threads>>>(
            x.data_ptr<float>(),
            depthwise_weight.data_ptr<float>(),
            depthwise_bias.defined() ? depthwise_bias.data_ptr<float>() : nullptr,
            depthwise_output.data_ptr<float>(),
            batch, in_channels,
            in_h, in_w,
            out_h, out_w,
            k, stride, padding, dilation);
    }));
    
    int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
    
    dim3 threadsPoint(TILE_SIZE, TILE_SIZE);
    dim3 blocksPoint((out_w + TILE_SIZE - 1) / TILE_SIZE,
                     (out_h + TILE_SIZE - 1) / TILE_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
        pointwise_conv2d_kernel_stride<<<blocksPoint, threadsPoint>>>(
            depthwise_output.data_ptr<float>(),
            pointwise_weight.data_ptr<float>(),
            pointwise_bias.defined() ? pointwise_bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch,
            in_channels,
            out_channels,
            out_h, out_w);
    }));
    
    return output;
}

at::Tensor toTensor(const py::object& obj) {
    if (obj.is_none()) return at::Tensor();
    try {
        return obj.cast<at::Tensor>();
    } catch (const py::cast_error& e) {
        if (py::hasattr(obj, "data")) {
            return obj.attr("data").cast<at::Tensor>();
        }
        throw std::runtime_error("Expected a torch Tensor or Parameter.");
    }
}

at::Tensor forward_wrapper(py::object x_obj,
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