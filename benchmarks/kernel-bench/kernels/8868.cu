__device__ __forceinline__ int gcd(int a, int b) {
  while(b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

__device__ __forceinline__ int my_min(int a, int b) {
  return a < b ? a : b;
}

__global__ void conv_transpose2d_kernel_optimized(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int in_channels_per_group,
    const int out_channels_per_group) {

    // Process 4 elements per thread
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * out_channels * out_h * out_w;
    
    #pragma unroll
    for (int elem = 0; elem < 4; elem++) {
        int index = thread_id * 4 + elem;
        if (index >= total_elements) break;

        int ow = index % out_w;
        int tmp = index / out_w;
        int oh = tmp % out_h;
        tmp = tmp / out_h;
        int oc = tmp % out_channels;
        int n = tmp / out_channels;

        float out_val = __ldg(&bias[oc]);
        int g = oc / out_channels_per_group;

        int candidate_h = oh + pad_h;
        int candidate_w = ow + pad_w;

        int offset_kh = -1;
        int mod_h = candidate_h % stride_h;
        #pragma unroll
        for (int k = 0; k < stride_h; k++) {
            if ((k * dilation_h) % stride_h == mod_h) {
                offset_kh = k;
                break;
            }
        }

        int offset_kw = -1;
        int mod_w = candidate_w % stride_w;
        #pragma unroll
        for (int k = 0; k < stride_w; k++) {
            if ((k * dilation_w) % stride_w == mod_w) {
                offset_kw = k;
                break;
            }
        }

        int step_kh = stride_h / gcd(stride_h, dilation_h);
        int step_kw = stride_w / gcd(stride_w, dilation_w);
        int kh_end = my_min(kernel_h, candidate_h / dilation_h + 1);
        int kw_end = my_min(kernel_w, candidate_w / dilation_w + 1);

        #pragma unroll 4
        for (int kh = offset_kh; kh >= 0 && kh < kh_end; kh += step_kh) {
            int ih = (candidate_h - kh * dilation_h) / stride_h;
            if (ih >= 0 && ih < in_h) {
                #pragma unroll 4
                for (int kw = offset_kw; kw >= 0 && kw < kw_end; kw += step_kw) {
                    int iw = (candidate_w - kw * dilation_w) / stride_w;
                    if (iw >= 0 && iw < in_w) {
                        #pragma unroll 4
                        for (int c = g * in_channels_per_group; c < (g + 1) * in_channels_per_group; c++) {
                            int x_idx = ((n * in_channels + c) * in_h + ih) * in_w + iw;
                            int w_idx = ((c * out_channels_per_group + (oc - g * out_channels_per_group)) 
                                      * kernel_h + kh) * kernel_w + kw;
                            out_val += __ldg(&x[x_idx]) * __ldg(&weight[w_idx]);
                        }
                    }
                }
            }
        }
        output[((n * out_channels + oc) * out_h + oh) * out_w + ow] = out_val;
    }
}