#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define TILE_SIZE 16

__global__ void conv_transpose3d_coalesced_kernel(
    const float* input,
    const float* weights,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int groups) {
    
    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_weights[TILE_SIZE][TILE_SIZE];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int channels_per_group = in_channels / groups;
    
    // Calculate output dimensions
    const int out_depth = (in_depth - 1) * stride_d - 2 * pad_d + kernel_d;
    const int out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_h;
    const int out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_w;
    
    // Ensure coalesced memory access by having threads in a warp process consecutive elements
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Process elements in a coalesced manner
    for (int n = 0; n < batch_size; n++) {
        for (int g = 0; g < groups; g++) {
            const int group_start = g * channels_per_group;
            
            for (int oc = warp_id; oc < out_channels/groups; oc += blockDim.x/WARP_SIZE) {
                for (int ic = lane_id; ic < channels_per_group; ic += WARP_SIZE) {
                    // Load input and weights into shared memory in a coalesced manner
                    #pragma unroll
                    for (int i = 0; i < TILE_SIZE; i += WARP_SIZE) {
                        if (tid + i < TILE_SIZE * TILE_SIZE) {
                            const int row = (tid + i) / TILE_SIZE;
                            const int col = (tid + i) % TILE_SIZE;
                            if (row < in_depth && col < in_height) {
                                shared_input[row][col] = input[n * in_channels * in_depth * in_height * in_width +
                                                             (group_start + ic) * in_depth * in_height * in_width +
                                                             row * in_height * in_width + col];
                            }
                        }
                    }
                    __syncthreads();
                    
                    // Compute output values with coalesced writes
                    #pragma unroll
                    for (int od = 0; od < out_depth; od++) {
                        const int id = od / stride_d;
                        for (int oh = 0; oh < out_height; oh++) {
                            const int ih = oh / stride_h;
                            #pragma unroll
                            for (int ow = lane_id; ow < out_width; ow += WARP_SIZE) {
                                const int iw = ow / stride_w;
                                float sum = 0.0f;
                                
                                #pragma unroll
                                for (int kd = 0; kd < kernel_d; kd++) {
                                    for (int kh = 0; kh < kernel_h; kh++) {
                                        for (int kw = 0; kw < kernel_w; kw++) {
                                            if (id + kd < in_depth && ih + kh < in_height && iw + kw < in_width) {
                                                sum += shared_input[id + kd][ih + kh] *
                                                       weights[((group_start + oc) * channels_per_group + ic) *
                                                               kernel_d * kernel_h * kernel_w +
                                                               kd * kernel_h * kernel_w +
                                                               kh * kernel_w + kw];
                                            }
                                        }
                                    }
                                }
                                
                                // Coalesced write to output
                                if (ow < out_width) {
                                    atomicAdd(&output[n * out_channels * out_depth * out_height * out_width +
                                                    (group_start + oc) * out_depth * out_height * out_width +
                                                    od * out_height * out_width +
                                                    oh * out_width + ow],
                                            sum);
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(*bias);
    }
    
    auto input_size = x.sizes();
    auto weight_size = weight.sizes();
    
    // Calculate output dimensions
    const int batch_size = input_size[0];
    const int in_channels = input_size[1];
    const int out_channels = weight_size[1] * groups;
    
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((batch_size * out_channels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    auto output = torch::zeros_like(x);
    
    conv_transpose3d_coalesced_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_size[2],
        input_size[3],
        input_size[4],
        weight_size[2],
        weight_size[3],
        weight_size[4],
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        groups);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed Conv3D forward (CUDA)");
}