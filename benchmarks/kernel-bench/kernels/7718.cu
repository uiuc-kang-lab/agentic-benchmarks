#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define TILE_DIM 16

template<typename scalar_t>
__forceinline__ __device__
scalar_t load_input(const scalar_t* ptr, int idx) {
    return __ldg(ptr + idx);
}

__global__ void conv3d_coalesced_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
    __shared__ float weight_shared[TILE_DIM][TILE_DIM];
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int batch_id = blockIdx.y;
    const int oc_base = blockIdx.x * TILE_DIM;
    
    // Pre-compute spatial dimensions
    const int spatial_size = out_depth * out_height * out_width;
    const int spatial_stride = out_height * out_width;
    
    // Process output points in a coalesced manner within each warp
    for (int spatial_idx = warp_id * WARP_SIZE + lane_id; 
         spatial_idx < spatial_size; 
         spatial_idx += blockDim.x) {
        
        const int od = spatial_idx / spatial_stride;
        const int remain = spatial_idx % spatial_stride;
        const int oh = remain / out_width;
        const int ow = remain % out_width;
        
        // Compute input base coordinates
        const int id_base = od * stride - padding;
        const int ih_base = oh * stride - padding;
        const int iw_base = ow * stride - padding;
        
        // Process output channels in tiles
        for (int oc_tile = 0; oc_tile < TILE_DIM && oc_base + oc_tile < out_channels; oc_tile++) {
            float sum = 0.0f;
            
            // Load weights into shared memory in a coalesced manner
            #pragma unroll 4
            for (int ic = 0; ic < in_channels; ic++) {
                for (int k = threadIdx.x; k < kernel_d * kernel_h * kernel_w; k += blockDim.x) {
                    const int kd = k / (kernel_h * kernel_w);
                    const int kh = (k % (kernel_h * kernel_w)) / kernel_w;
                    const int kw = k % kernel_w;
                    weight_shared[threadIdx.x % TILE_DIM][k / TILE_DIM] = 
                        weight[((oc_base + oc_tile) * in_channels + ic) * kernel_d * kernel_h * kernel_w + k];
                }
                __syncthreads();
                
                #pragma unroll 2
                for (int kd = 0; kd < kernel_d; kd++) {
                    const int id = id_base + kd * dilation;
                    if (id >= 0 && id < in_depth) {
                        #pragma unroll 2
                        for (int kh = 0; kh < kernel_h; kh++) {
                            const int ih = ih_base + kh * dilation;
                            if (ih >= 0 && ih < in_height) {
                                #pragma unroll 4
                                for (int kw = 0; kw < kernel_w; kw++) {
                                    const int iw = iw_base + kw * dilation;
                                    if (iw >= 0 && iw < in_width) {
                                        const int in_idx = ((batch_id * in_channels + ic) * in_depth + id) * 
                                                         in_height * in_width + ih * in_width + iw;
                                        const int w_idx = (kd * kernel_h + kh) * kernel_w + kw;
                                        sum += load_input(input, in_idx) * 
                                              weight_shared[oc_tile][w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
                __syncthreads();
            }
            
            if (bias != nullptr) {
                sum += bias[oc_base + oc_tile];
            }
            
            // Write output in a coalesced manner
            const int out_idx = ((batch_id * out_channels + (oc_base + oc_tile)) * out_depth + od) * 
                               out_height * out_width + oh * out_width + ow;
            output[out_idx] = sum;
        }
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
    
    dim3 grid((out_channels + TILE_DIM - 1) / TILE_DIM, batch_size);
    dim3 block(BLOCK_SIZE);
    
    conv3d_coalesced_kernel<<<grid, block>>>(
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
    m.def("forward", &forward, "Coalesced 3D convolution forward (CUDA)");
}