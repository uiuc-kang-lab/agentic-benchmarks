#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Tile sizes for shared memory optimization
#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 11

__global__ void conv_transpose2d_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int chunkN, int C_in, int H, int W,
    int C_out, int K, int stride,
    int padding, int H_out, int W_out) {

    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + TILE_SIZE * TILE_SIZE;

    // Block handles a tile of output
    int tile_row = blockIdx.y * TILE_SIZE;
    int tile_col = blockIdx.z * TILE_SIZE;
    int n = blockIdx.x % chunkN;
    int c_out = threadIdx.z;

    // Load input tile into shared memory
    if (threadIdx.z == 0) {
        int in_row = (tile_row + threadIdx.y + padding) / stride;
        int in_col = (tile_col + threadIdx.x + padding) / stride;
        if (in_row >= 0 && in_row < H && in_col >= 0 && in_col < W) {
            for (int c = 0; c < C_in; c++) {
                int in_idx = n * (C_in * H * W) + c * (H * W) + in_row * W + in_col;
                shared_input[threadIdx.y * TILE_SIZE + threadIdx.x] = input[in_idx];
            }
        }
    }
    __syncthreads();

    // Load weights into shared memory
    if (threadIdx.x < K && threadIdx.y < K) {
        for (int c = 0; c < C_in; c++) {
            int w_idx = c * (C_out * K * K) + c_out * (K * K) + threadIdx.y * K + threadIdx.x;
            shared_weight[threadIdx.y * K + threadIdx.x] = weight[w_idx];
        }
    }
    __syncthreads();

    // Compute output for this thread's position
    int out_row = tile_row + threadIdx.y;
    int out_col = tile_col + threadIdx.x;
    
    if (out_row < H_out && out_col < W_out) {
        float sum = 0.0f;
        
        for (int c = 0; c < C_in; c++) {
            for (int ky = 0; ky < K; ky++) {
                for (int kx = 0; kx < K; kx++) {
                    int in_row = (out_row + padding - ky) / stride;
                    int in_col = (out_col + padding - kx) / stride;
                    
                    if (in_row >= 0 && in_row < H && in_col >= 0 && in_col < W &&
                        (out_row + padding - ky) % stride == 0 &&
                        (out_col + padding - kx) % stride == 0) {
                        
                        int in_tile_idx = (in_row % TILE_SIZE) * TILE_SIZE + (in_col % TILE_SIZE);
                        int w_idx = ky * K + kx;
                        sum += shared_input[in_tile_idx] * shared_weight[w_idx];
                    }
                }
            }
        }

        int out_idx = n * (C_out * H_out * W_out) + 
                      c_out * (H_out * W_out) +
                      out_row * W_out + out_col;
        atomicAdd(&output[out_idx], sum);
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

    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "Inputs must be contiguous");

    auto x_sizes = x.sizes();
    int N = x_sizes[0];
    int C_in = x_sizes[1];
    int H = x_sizes[2];
    int W = x_sizes[3];
    
    auto w_sizes = weight.sizes();
    int C_out = w_sizes[1];
    int K = w_sizes[2];

    int H_out = (H - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W - 1) * stride - 2 * padding + K + output_padding;

    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    const int nstreams = 4;
    std::vector<cudaStream_t> streams(nstreams);
    for (int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int chunk = (N + nstreams - 1) / nstreams;
    
    dim3 block(TILE_SIZE, TILE_SIZE, C_out);
    dim3 grid(chunk, 
              (H_out + TILE_SIZE - 1) / TILE_SIZE,
              (W_out + TILE_SIZE - 1) / TILE_SIZE);
              
    int shared_mem_size = (TILE_SIZE * TILE_SIZE + K * K) * sizeof(float);

    for (int i = 0; i < nstreams; i++) {
        int start_n = i * chunk;
        int chunk_size = std::min(chunk, N - start_n);
        if (chunk_size <= 0) continue;

        conv_transpose2d_tiled_kernel<<<grid, block, shared_mem_size, streams[i]>>>(
            x.data_ptr<float>() + start_n * C_in * H * W,
            weight.data_ptr<float>(),
            output.data_ptr<float>() + start_n * C_out * H_out * W_out,
            chunk_size, C_in, H, W, C_out, K, stride, padding, H_out, W_out
        );
    }

    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d optimized forward (CUDA)");
}