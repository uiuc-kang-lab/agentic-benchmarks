#include <torch/extension.h>

template <int BLOCK_SIZE = 16, int TILE_SIZE = 8>
__global__ void convTranspose2dSharedKernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    // Shared memory for weight tiles
    __shared__ float weight_shared[TILE_SIZE][TILE_SIZE];
    
    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Output position
    const int out_x = bx * BLOCK_SIZE + tx;
    const int out_y = by * BLOCK_SIZE + ty;
    const int b = bz / out_channels;
    const int oc = bz % out_channels;
    
    float sum = 0.0f;
    
    // Loop over input channels in tiles
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kt = 0; kt < kernel_size; kt += TILE_SIZE) {
            for (int kw = 0; kw < kernel_size; kw += TILE_SIZE) {
                // Collaborative loading of weight tiles into shared memory
                if (tx < TILE_SIZE && ty < TILE_SIZE) {
                    if ((kt + tx) < kernel_size && (kw + ty) < kernel_size) {
                        weight_shared[tx][ty] = weight[
                            ((ic * out_channels + oc) * kernel_size + (kt + tx)) * kernel_size + (kw + ty)
                        ];
                    } else {
                        weight_shared[tx][ty] = 0.0f;
                    }
                }
                __syncthreads();
                
                // Process the tile
                if (out_x < H_out && out_y < W_out) {
                    for (int i = 0; i < TILE_SIZE; i++) {
                        if (kt + i >= kernel_size) break;
                        for (int j = 0; j < TILE_SIZE; j++) {
                            if (kw + j >= kernel_size) break;
                            
                            // Calculate input position
                            int h_in = (out_x + padding - (kt + i));
                            int w_in = (out_y + padding - (kw + j));
                            
                            if (h_in % stride == 0 && w_in % stride == 0) {
                                h_in /= stride;
                                w_in /= stride;
                                
                                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                    const int input_idx = ((b * in_channels + ic) * H_in + h_in) * W_in + w_in;
                                    sum += input[input_idx] * weight_shared[i][j];
                                }
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    
    // Write output
    if (out_x < H_out && out_y < W_out) {
        const int output_idx = ((b * out_channels + oc) * H_out + out_x) * W_out + out_y;
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[output_idx] = sum;
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    TORCH_CHECK(groups == 1, "Groups != 1 not supported");
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(1);
    
    const int H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, H_out, W_out}, x.options());
    
    constexpr int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (H_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (W_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size * out_channels
    );
    
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    
    convTranspose2dSharedKernel<BLOCK_SIZE, 8><<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        H_in,
        W_in,
        H_out,
        W_out,
        kernel_size,
        stride,
        padding
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA)");
}