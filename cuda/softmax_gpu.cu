#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

void softmax_forward_cpu(float *out, const float *inp, int N, int C) {
    for (int i = 0; i < N; i++) {
        const float *inp_row = inp + i * C;
        float *out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        float sum = 0.f;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        float norm = 1.f / sum;
        for (int j = 0; j < C; j++) {
            out_row[j] *= norm;
        }
    }
}

// CUDA kernel
__global__ void softmax_forward_kernel1(float *out, const float *inp, int N,
                                        int C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    const float *inp_row = inp + i * C;
    float *out_row = out + i * C;

    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
      if (inp_row[j] > maxval) {
        maxval = inp_row[j];
      }
    }
    float sum = 0.f;
    for (int j = 0; j < C; j++) {
      out_row[j] = expf(inp_row[j] - maxval);
      sum += out_row[j];
    }
    for (int j = 0; j < C; j++) {
      out_row[j] /= (float)sum;
    }
  }
}

// Function to compare results
bool compare_results(const float *cpu, const float *gpu, int N, int C,
                     float epsilon = 1e-3f) {
  for (int i = 0; i < N * C; ++i) {
    if (fabs(cpu[i] - gpu[i]) > epsilon) {
      std::cout << "Difference at index " << i << ": CPU=" << cpu[i]
                << ", GPU=" << gpu[i] << ", diff=" << fabs(cpu[i] - gpu[i])
                << std::endl;
      return false;
    }
  }
  return true;
}

int main() {
    int N = 32;
    int C = 4096;

    size_t num_elements = N * C;
    float *inp = (float *)malloc(num_elements * sizeof(float));
    float *out_cpu = (float *)malloc(num_elements * sizeof(float));


    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            inp[n * C + c] = float(c);
        }
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    softmax_forward_cpu(out_cpu, inp, N, C);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;

    free(inp);
    free(out_cpu);

    return 0;
}
