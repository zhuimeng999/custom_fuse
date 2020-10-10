//
// Created by lucius on 10/10/20.
//

#include "depth_filter_kernel.cuh"

// a simple kernel that simply increments each array element by b
__global__ void kernelAddConstant(int *g_a, const int b)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_a[idx] += b;
}

