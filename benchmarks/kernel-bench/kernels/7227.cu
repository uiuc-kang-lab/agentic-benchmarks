#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<bool USE_SHARED_MEM>
__global__ void conv2d_hybrid_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
    
    int oc = blockIdx.z % out_channels;
    int n = blockIdx.z / out_channels;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row >= out_h || out_col >= out_w) return;

    extern __shared__ float sh_weight[];
    if (USE_SHARED_MEM) {
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int filter_elems = in_channels * kernel_size * kernel_size;
        
        for (int idx = tid; idx < filter_elems; idx += blockDim.x * blockDim.y) {
            int ic = idx / (kernel_size * kernel_size);
            int rem = idx % (kernel_size * kernel_size);
            int ki = rem / kernel_size;
            int kj = rem % kernel_size;
            sh_weight[idx] = weight[oc * filter_elems + idx];
        }
        __syncthreads();
    }

    float sum = 0.0f;
    #pragma unroll
    for (int ic = 0; ic < in_channels; ++ic) {
        #pragma unroll
        for (int ki = 0; ki < kernel_size; ++ki) {
            int in_row = out_row * stride - padding + ki;
            if (in_row < 0 || in_row >= in_h) continue;
            
            #pragma unroll
            for (int kj = 0; kj < kernel_size; ++kj) {
                int in_col = out_col * stride - padding + kj;
                if (in_col < 0 || in_col >= in_w) continue;
                
                int input_idx = n * (in_channels * in_h * in_w) + 
                               ic * (in_h * in_w) + 
                               in_row * in_w + in_col;
                               
                float weight_val;
                if (USE_SHARED_MEM) {
                    int filter_idx = ic * (kernel_size * kernel_size) + 
                                   ki * kernel_size + kj;
                    weight_val = sh_weight[filter_idx];
                } else {
                    int weight_idx = oc * (in_channels * kernel_size * kernel_size) +
                                   ic * (kernel_size * kernel_size) + 
                                   ki * kernel_size + kj;
                    weight_val = __ldg(&weight[weight_idx]);
                }
                
                sum += input[input_idx] * weight_val;
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    int out_idx = n * (out_channels * out_h * out_w) + 
                  oc * (out_h * out_w) + 
                  out_row * out_w + out_col;
    output[out_idx] = sum;
}

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
        return torch::conv2d(x, weight, 
            bias.has_value() ? bias.value() : torch::Tensor(),
            {stride, stride}, {padding, padding},
            {dilation, dilation}, groups);
    }

    auto dims = x.sizes();
    int batch_size = dims[0];
    int in_channels = dims[1];
    int in_h = dims[2];
    int in_w = dims[3];
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, 
                             x.options());

    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x,
              (out_h + block.y - 1) / block.y,
              batch_size * out_channels);

    size_t shared_mem_size = in_channels * kernel_size * kernel_size * sizeof(float);
    bool use_shared_mem = (shared_mem_size <= 48 * 1024);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (use_shared_mem) {
        conv2d_hybrid_kernel<true><<<grid, block, shared_mem_size, stream>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(),
            bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size, in_channels, in_h, in_w,
            out_channels, kernel_size, out_h, out_w,
            stride, padding);
    } else {
        conv2d_hybrid_kernel<false><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(),
            bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size, in_channels, in_h, in_w,
            out_channels, kernel_size, out_h, out_w,
            stride, padding);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid CUDA convolution forward");
}