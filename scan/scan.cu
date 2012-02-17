#include <iostream>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

using std::cout;
using std::endl;

using std::string;

template<typename T>
void print1D(string const& prefix, int count, T const& value, string const& infix, string const& postfix = "")
{
  cout << prefix;
  int i;
  for (i = 0; i < count - 1; ++i)
  {
    cout << value[i] << infix;
  }
  cout << value[i] << postfix << endl;
}

template<typename T>
void printArray(int count, T const& array)
{
  print1D("[", count, array, ", ", "]");
}

#define BLOCK_SIZE 4
#define LOG(value) ((int)log2((double)(value)))
#define ODD(value) ((value)&0x01)
#define DOUBLE(value) ((value) << 1)
#define HALF(value) ((value) >> 1)

template<typename T>
__global__ void prefixScan(int count, T *values)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ T block_values[BLOCK_SIZE];
  block_values[threadIdx.x] = values[id];
  __syncthreads();

  int i, id1 = id + 1;
  for ( i = 2; i < count; i = DOUBLE(i) )
  {
    if (id1 % i == 0)
    {
      block_values[threadIdx.x] += block_values[threadIdx.x - HALF(i)];
    }
  }
  __syncthreads();

  for ( i = HALF(BLOCK_SIZE); i >= 2; i = HALF(i) )
  {
    if (id1 % i == 0 && id1 != BLOCK_SIZE)
    {
      block_values[threadIdx.x + HALF(i)] += block_values[threadIdx.x]; 
    }
  }
  __syncthreads();

  values[id] = block_values[threadIdx.x];
}

int main()
{
  int hostValues [] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  const int COUNT = 16;
  const int height = LOG(COUNT/BLOCK_SIZE);

  cudaDeviceReset();

  printf("Height: %d\n", height);

  const int size = sizeof(int) * COUNT;

  int *deviceValues;
  cudaMalloc(&deviceValues, size);

  cudaMemcpy(deviceValues, hostValues, size, cudaMemcpyHostToDevice);

  prefixScan<<<4,4>>>(COUNT, deviceValues);

  cudaMemcpy(hostValues, deviceValues, size, cudaMemcpyDeviceToHost);

  printArray(COUNT, hostValues);

  cudaFree(deviceValues);

  return 0;
}
