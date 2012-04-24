#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cstddef>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "types.hpp"
#include "problem.hpp"

using std::cout;
using std::endl;

using std::string;

texture<float, cudaTextureType2D, cudaReadModeElementType> textureRef;

void randomFill(Data* values, size_t length, Data rand_max)
{
  Data quotient = ((Data)RAND_MAX) / rand_max;
  for ( size_t i = 0; i < length; ++i )
  {
    values[i] = ((Data)rand()) / quotient;
  }
}

template<size_t size, typename type>
void print(type * const array)
{
  if ( size == 0 )
  {
    cout << "[]" << endl;
    return;
  }

  cout << '[';
  size_t lastIndex = size - 1;
  for ( size_t i = 0 ; i < lastIndex ; ++i )
  {
    cout << array[i] << ", ";
  }
  cout << array[lastIndex] << ']' << endl;
}

int main()
{

  Data e = 0.1;
  size_t N = 3;
  size_t M = 3;
  Data d[3] = { 7, 8, 9 };
  Data b[3] = { 4, 5, 6 };
  Data c[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

  print<9>(c);

  Problem initialProblem(e, N, M, d, c, b);
  SpecialProblem specialProblem(initialProblem);

  print<9>(c);

  print<3>(specialProblem.objective);
  print<9>(specialProblem.constraintMatrix);

/*
  int N = 200;
  int BLOCK = 10;
  int SIZE = sizeof(data) * N;

  data* hostValues = (data*)malloc(SIZE);
  randomFill(hostValues, N, 100.0f);

  data* deviceValues;
  cudaMalloc(&deviceValues, SIZE);

  cudaMemcpy(deviceValues, hostValues, SIZE, cudaMemcpyHostToDevice);
  exponent<<<N/BLOCK, BLOCK>>>(deviceValues);
  cudaMemcpy(hostValues, deviceValues, SIZE, cudaMemcpyDeviceToHost);

*/

  return 0;
}
