#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define TILE_SIZE 8
#define MAX_SHARED_SIZE 48*1024  // 48KB shared memory per block on H100

template<typename scalar_t>
__global__ void conv3d_optimized_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation
) {
    extern __shared__ float shared_mem[];
    
    // Shared memory layout:
    // First part: weights for current output channel
    // Second part: input tile
    float* shared_weights = shared_mem;
    float* shared_input = &shared_mem[kernel_d * kernel_h * kernel_w];
    
    const int tid = threadIdx.x;
    const int oc = blockIdx.x;
    const int batch_id = blockIdx.y;
    
    // Load weights into shared memory for current input channel
    const int weights_per_channel = kernel_d * kernel_h * kernel_w;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int i = tid; i < weights_per_channel; i += blockDim.x) {
            const int kd = i / (kernel_h * kernel_w);
            const int remainder = i % (kernel_h * kernel_w);
            const int kh = remainder / kernel_w;
            const int kw = remainder % kernel_w;
            
            shared_weights[i] = weight[((oc * in_channels + ic) * kernel_d + kd) * 
                                      kernel_h * kernel_w + kh * kernel_w + kw];
        }
        __syncthreads();
    
    // Process output elements
    const int total_elements = out_depth * out_height * out_width;
    for (int idx = tid; idx < total_elements; idx += blockDim.x) {
        const int od = idx / (out_height * out_width);
        const int tmp = idx % (out_height * out_width);
        const int oh = tmp / out_width;
        const int ow = tmp % out_width;
        
        float sum = 0.0f;
        
        // Process input channels
        for (int ic = 0; ic < in_channels; ++ic) {
            // Load input tile into shared memory
            const int tile_size = TILE_SIZE * TILE_SIZE * TILE_SIZE;
            const int id_start = od * stride - padding;
            const int ih_start = oh * stride - padding;
            const int iw_start = ow * stride - padding;
            
            // Load input data for current tile
            for (int t = tid; t < tile_size; t += blockDim.x) {
                const int td = t / (TILE_SIZE * TILE_SIZE);
                const int th = (t / TILE_SIZE) % TILE_SIZE;
                const int tw = t % TILE_SIZE;
                
                const int id = id_start + td;
                const int ih = ih_start + th;
                const int iw = iw_start + tw;
                
                if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    shared_input[t] = input[((batch_id * in_channels + ic) * in_depth + id) *
                                          in_height * in_width + ih * in_width + iw];
                } else {
                    shared_input[t] = 0.0f;
                }
            }
            __syncthreads();
            
            // Compute convolution using shared memory
            #pragma unroll 2
            for (int kd = 0; kd < kernel_d; ++kd) {
                const int id = od * stride - padding + kd * dilation;
                if (id >= 0 && id < in_depth) {
                    #pragma unroll 2
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        const int ih = oh * stride - padding + kh * dilation;
                        if (ih >= 0 && ih < in_height) {
                            #pragma unroll 4
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                const int iw = ow * stride - padding + kw * dilation;
                                if (iw >= 0 && iw < in_width) {
                                    const int w_idx = (kd * kernel_h + kh) * kernel_w + kw;
                                    const int in_idx = (id % TILE_SIZE) * TILE_SIZE * TILE_SIZE +
                                                     (ih % TILE_SIZE) * TILE_SIZE + (iw % TILE_SIZE);
                                    sum += shared_input[in_idx] * shared_weights[w_idx];
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        const int output_idx = ((batch_id * out_channels + oc) * out_depth + od) *
                              out_height * out_width + oh * out_width + ow;
        output[output_idx] = sum;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    TORCH_CHECK(groups == 1, "Only groups=1 is supported");
    auto bias = bias_opt.value_or(at::Tensor());
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    
    const int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    const int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width},
                           input.options());
    
    // Calculate shared memory size
    const int shared_mem_size = (kernel_d * kernel_h * kernel_w + TILE_SIZE * TILE_SIZE * TILE_SIZE) * sizeof(float);
    TORCH_CHECK(shared_mem_size <= MAX_SHARED_SIZE, "Shared memory size exceeds device limit");
    
    dim3 grid(out_channels, batch_size);
    const int threads = BLOCK_SIZE;
    
    conv3d_optimized_kernel<<<grid, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D convolution forward with shared memory (CUDA)");
}