#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool2d_stream_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int C, int H, int W,
    int outH, int outW,
    int kernel_size, int stride, int padding
) {
    extern __shared__ char shared_memory[];
    scalar_t* tile = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int TILE_SIZE = 32;  // Tile size for shared memory
    const int tile_w = TILE_SIZE + kernel_size - 1;
    const int tile_h = TILE_SIZE + kernel_size - 1;
    
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    int bx = blockIdx.x % ((outW + TILE_SIZE - 1) / TILE_SIZE);
    int by = (blockIdx.x / ((outW + TILE_SIZE - 1) / TILE_SIZE)) % ((outH + TILE_SIZE - 1) / TILE_SIZE);
    int c = blockIdx.x / (((outW + TILE_SIZE - 1) / TILE_SIZE) * ((outH + TILE_SIZE - 1) / TILE_SIZE));
    
    if (c >= C) return;
    
    // Calculate input tile origin
    int tile_x_start = bx * TILE_SIZE * stride - padding;
    int tile_y_start = by * TILE_SIZE * stride - padding;
    
    // Load input tile into shared memory
    for (int i = ty; i < tile_h; i += (blockDim.x / TILE_SIZE)) {
        for (int j = tx; j < tile_w; j += TILE_SIZE) {
            int h_in = tile_y_start + i;
            int w_in = tile_x_start + j;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                tile[i * tile_w + j] = input[(c * H + h_in) * W + w_in];
            } else {
                tile[i * tile_w + j] = 0;
            }
        }
    }
    
    __syncthreads();
    
    // Process output elements
    int h_out = by * TILE_SIZE + ty;
    int w_out = bx * TILE_SIZE + tx;
    
    if (h_out < outH && w_out < outW && c < C) {
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;
        
        // Calculate local indices in the tile
        int local_h_start = h_start - tile_y_start;
        int local_w_start = w_start - tile_x_start;
        
        scalar_t sum_val = scalar_t(0);
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int local_h = local_h_start + i;
                int local_w = local_w_start + j;
                sum_val += tile[local_h * tile_w + local_w];
            }
        }
        
        output[(c * outH + h_out) * outW + w_out] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}

torch::Tensor avg_pool2d_forward(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor.");
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto x_cont = x.contiguous();
    auto out = torch::empty({N, C, outH, outW}, x.options());

    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    const int threads = 128;
    const int elements_per_sample = C * outH * outW;
    const int blocks = (elements_per_sample + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward", [&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();

        for (int n = 0; n < N; ++n) {
            int stream_id = n % num_streams;
            const scalar_t* sample_input = input_data + n * C * H * W;
            scalar_t* sample_output = output_data + n * C * outH * outW;
            
            avg_pool2d_stream_kernel<<<blocks, threads, 0, streams[stream_id]>>>(
                sample_input, sample_output,
                C, H, W, outH, outW,
                kernel_size, stride, padding
            );
        }
    });

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward (CUDA)");
}