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

#define BLOCK_SIZE 8
#define DOUBLE(value) ((value) << 1)
#define HALF(value) ((value) >> 1)
#define GLOBAL_ID (blockIdx.x * blockDim.x + threadIdx.x)

template<typename T>
__global__ void scan(int count, int offset, T *values)
{
	int id = GLOBAL_ID;
  int offset_id = id * offset + (offset == 1 ? 0 : offset - 1);

  __shared__ T block_values[BLOCK_SIZE];
  block_values[threadIdx.x] = values[offset_id];
  __syncthreads();

  int i, id1 = id + 1;
  for ( i = 2; i <= count; i = DOUBLE(i) )
  {
    if (id1 % i == 0)
    {
      block_values[threadIdx.x] += block_values[threadIdx.x - HALF(i)];
    }
    __syncthreads();
  }

  for ( i = HALF(BLOCK_SIZE); i >= 2; i = HALF(i) )
  {
    if (id1 % i == 0 && id1 != BLOCK_SIZE)
    {
      block_values[threadIdx.x + HALF(i)] += block_values[threadIdx.x]; 
    }
    __syncthreads();
  }

  values[offset_id] = block_values[threadIdx.x];
}

int main()
{
  int hostValues [] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  const int COUNT = 16;

  cudaDeviceReset();

  const int size = sizeof(int) * COUNT;

  int *deviceValues;
  cudaMalloc(&deviceValues, size);

  cudaMemcpy(deviceValues, hostValues, size, cudaMemcpyHostToDevice);

  scan<<<2,BLOCK_SIZE>>>(COUNT, 1, deviceValues);

  scan<<<1,2>>>(COUNT/BLOCK_SIZE, BLOCK_SIZE, deviceValues);

  cudaMemcpy(hostValues, deviceValues, size, cudaMemcpyDeviceToHost);

  printArray(COUNT, hostValues);

  cudaFree(deviceValues);

  return 0;
}
