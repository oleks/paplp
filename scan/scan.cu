#include <iostream>
#include <stdio.h>

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

__global__ void prefixScan(int *values)
{
	values[blockIdx.x * blockDim.x + threadIdx.x] = threadIdx.x;
}

int main()
{
  int hostValues [] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  const int COUNT = 16;
  const int size = sizeof(int) * COUNT;

  int *deviceValues;
  cudaMalloc(&deviceValues, size);

  cudaMemcpy(deviceValues, hostValues, size, cudaMemcpyHostToDevice);

  prefixScan<<<4,4>>>(deviceValues);

  cudaMemcpy(hostValues, deviceValues, size, cudaMemcpyDeviceToHost);

  printArray(COUNT, hostValues);

  cudaFree(deviceValues);

  return 0;
}
