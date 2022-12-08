#include <stdio.h>

#define NBLOCKS  2
#define TPERBLK  1

#define NTHREADS (NBLOCKS * TPERBLK)

void errCheck()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error %d: %s\n", err, cudaGetErrorString(err));
        exit(1);
    }
}


// @@@code  {inter block atomic race}   {cuda:bbatomic}
__device__ int flag = 0;
__device__ int lock = 0;

__global__ void kmain(unsigned int *data)   // @@@{
{
    while (atomicCAS(&lock, 0, 1) != 0)
    __threadfence();
    data[0]++;
    __threadfence();
    atomicExch(&lock, 0);
    while (atomicCAS(&lock, 0, 1) != 0)
    __threadfence();
    data[0]*=2;
    __threadfence();
    atomicExch(&lock, 0);
    while (atomicCAS(&lock, 0, 1) != 0)
    __threadfence();
    data[0]+=3;
    __threadfence();
    atomicExch(&lock, 0);
}                                           // @@@}

int main() 
{
    unsigned int *d_data;
    cudaMalloc(&d_data, sizeof(unsigned int));
    unsigned int *t = (unsigned int*)malloc(sizeof(unsigned int));
    t[0] = 0;
    cudaMemcpy(d_data, t, sizeof(unsigned int), cudaMemcpyHostToDevice);
    kmain<<<NBLOCKS,TPERBLK>>>(d_data);
    cudaMemcpy(t, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%u\n", t[0]);
    errCheck();
    return 0;
}
