#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

// Kernel with coalesced memory access for depthwise convolution
template <typename scalar_t>
__global__ void depthwiseConv2DCoalescedKernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {
    
    extern __shared__ char shared_mem[];
    scalar_t* shared_weights = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int nc = blockIdx.z;
    const int c = nc % in_channels;
    const int n = nc / in_channels;
    
    // Load kernel weights into shared memory
    if (tid < kernel_size * kernel_size) {
        shared_weights[tid] = w[c * kernel_size * kernel_size + tid];
    }
    __syncthreads();
    
    // Calculate base indices
    const int elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int total_elements = out_height * out_width;
    const int base_idx = blockIdx.y * elements_per_block;
    
    // Process multiple elements per thread
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int element_idx = base_idx + tid + i * BLOCK_SIZE;
        if (element_idx >= total_elements) continue;
        
        const int out_y = element_idx / out_width;
        const int out_x = element_idx % out_width;
        
        // Pre-compute input positions
        const int in_y_base = out_y * stride - padding;
        const int in_x_base = out_x * stride - padding;
        
        scalar_t sum = 0;
        
        // Unroll the kernel loops for better instruction-level parallelism
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            const int in_y = in_y_base + ky;
            const bool valid_y = (in_y >= 0 && in_y < in_height);
            
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                const int in_x = in_x_base + kx;
                
                if (valid_y && in_x >= 0 && in_x < in_width) {
                    const scalar_t input_val = x[((n * in_channels + c) * in_height + in_y) * in_width + in_x];
                    const scalar_t weight_val = shared_weights[ky * kernel_size + kx];
                    sum += input_val * weight_val;
                }
            }
        }
        
        // Add bias and write output
        if (element_idx < total_elements) {
            sum += b[c];
            out[((n * in_channels + c) * out_height + out_y) * out_width + out_x] = sum;
        }
    }
}

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups) {
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());
    
    const int total_elements = out_height * out_width;
    const int elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int num_blocks_y = (total_elements + elements_per_block - 1) / elements_per_block;
    
    dim3 grid(1, num_blocks_y, batch_size * in_channels);
    dim3 block(BLOCK_SIZE);
    
    const int shared_mem_size = kernel_size * kernel_size * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_coalesced", ([&] {
        depthwiseConv2DCoalescedKernel<scalar_t><<<grid, block, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            kernel_size,
            out_height,
            out_width,
            stride,
            padding
        );
    }));
    
    return out;
}

torch::Tensor forward_wrap(
    torch::Tensor x,
    torch::Tensor weight,
    pybind11::object bias_obj,
    int stride,
    int padding,
    int groups) {
    torch::Tensor bias;
    if (bias_obj.is_none()) {
        bias = torch::zeros({x.size(1)}, x.options());
    } else {
        bias = bias_obj.cast<torch::Tensor>();
    }
    return forward_impl(x, weight, bias, stride, padding, groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",
          &forward_wrap,
          "Coalesced memory access depthwise conv2d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("groups") = 1);
}