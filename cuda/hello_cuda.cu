# include <stdio.h>

__global__ void kernel() {
    printf("Hello from GPU!\n");
}

int main() {
    printf("Hello from CPU!\n");
    int a = 3;
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
