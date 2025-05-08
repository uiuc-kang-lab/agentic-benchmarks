#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

__global__ void transposed_conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_offset,
    const int batch_size,
    const int in_channels,
    const int in_d, const int in_h, const int in_w,
    const int out_channels,
    const int out_d, const int out_h, const int out_w,
    const int k_d, const int k_h, const int k_w,
    const int s_d, const int s_h, const int s_w,
    const int p_d, const int p_h, const int p_w,
    const int groups,
    const int channels_per_group_in,
    const int channels_per_group_out) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * out_channels * out_d * out_h * out_w;
    
    if (idx < total) {
        int tmp = idx;
        const int w_out = tmp % out_w; tmp /= out_w;
        const int h_out = tmp % out_h; tmp /= out_h;
        const int d_out = tmp % out_d; tmp /= out_d;
        const int oc = tmp % out_channels;
        const int n = tmp / out_channels + batch_offset;

        float sum = (bias != nullptr) ? bias[oc] : 0.0f;
        
        const int group = oc / channels_per_group_out;
        const int oc_in_group = oc % channels_per_group_out;
        
        const int d_base = d_out + p_d;
        const int h_base = h_out + p_h;
        const int w_base = w_out + p_w;

        #pragma unroll 4
        for (int kd = 0; kd < k_d; kd++) {
            const int tmp_d = d_base - kd;
            if (tmp_d % s_d != 0) continue;
            const int in_d_idx = tmp_d / s_d;
            if (in_d_idx < 0 || in_d_idx >= in_d) continue;

            #pragma unroll 4
            for (int kh = 0; kh < k_h; kh++) {
                const int tmp_h = h_base - kh;
                if (tmp_h % s_h != 0) continue;
                const int in_h_idx = tmp_h / s_h;
                if (in_h_idx < 0 || in_h_idx >= in_h) continue;

                #pragma unroll 4
                for (int kw = 0; kw < k_w; kw++) {
                    const int tmp_w = w_base - kw;
                    if (tmp_w % s_w != 0) continue;
                    const int in_w_idx = tmp_w / s_w;
                    if (in_w_idx < 0 || in_w_idx >= in_w) continue;

                    #pragma unroll 4
                    for (int ic = 0; ic < channels_per_group_in; ic++) {
                        const int in_channel = group * channels_per_group_in + ic;
                        const int input_idx = n * (in_channels * in_d * in_h * in_w) +
                                            in_channel * (in_d * in_h * in_w) +
                                            in_d_idx * (in_h * in_w) +
                                            in_h_idx * in_w + in_w_idx;
                        
                        const int weight_idx = in_channel * (channels_per_group_out * k_d * k_h * k_w) +
                                             oc_in_group * (k_d * k_h * k_w) +
                                             kd * (k_h * k_w) + kh * k_w + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias_opt.has_value()) {
        CHECK_INPUT(*bias_opt);
    }

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_d = x.size(2);
    const int in_h = x.size(3);
    const int in_w = x.size(4);
    
    const int k_d = weight.size(2);
    const int k_h = weight.size(3);
    const int k_w = weight.size(4);
    
    const int s_d = stride[0];
    const int s_h = stride[1];
    const int s_w = stride[2];
    
    const int p_d = padding[0];
    const int p_h = padding[1];
    const int p_w = padding[2];
    
    const int op_d = output_padding[0];
    const int op_h = output_padding[1];
    const int op_w = output_padding[2];

    const int out_d = (in_d - 1) * s_d - 2 * p_d + k_d + op_d;
    const int out_h = (in_h - 1) * s_h - 2 * p_h + k_h + op_h;
    const int out_w = (in_w - 1) * s_w - 2 * p_w + k_w + op_w;

    const int channels_per_group_out = weight.size(1);
    const int out_channels = channels_per_group_out * groups;
    const int channels_per_group_in = in_channels / groups;

    auto output = torch::zeros({batch, out_channels, out_d, out_h, out_w}, x.options());

    // Create CUDA streams
    cudaStream_t compute_stream, transfer_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&transfer_stream);

    // Calculate chunk size for batch processing
    const int chunk_size = 4;  // Process 4 samples at a time
    const int num_chunks = (batch + chunk_size - 1) / chunk_size;

    // Get raw pointers
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias_opt.has_value() ? (*bias_opt).data_ptr<float>() : nullptr;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int chunk_start = chunk * chunk_size;
        const int current_chunk_size = std::min(chunk_size, batch - chunk_start);
        
        const int elements_per_chunk = current_chunk_size * out_channels * out_d * out_h * out_w;
        const int threads = 256;
        const int blocks = (elements_per_chunk + threads - 1) / threads;

        // Launch kernel on compute stream
        transposed_conv3d_kernel<<<blocks, threads, 0, compute_stream>>>(
            x.data_ptr<float>(),
            weight_ptr,
            bias_ptr,
            output.data_ptr<float>(),
            chunk_start,
            current_chunk_size,
            in_channels,
            in_d, in_h, in_w,
            out_channels,
            out_d, out_h, out_w,
            k_d, k_h, k_w,
            s_d, s_h, s_w,
            p_d, p_h, p_w,
            groups,
            channels_per_group_in,
            channels_per_group_out);
    }

    // Cleanup streams
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(transfer_stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed Conv3D forward with stream overlap (CUDA)");
}