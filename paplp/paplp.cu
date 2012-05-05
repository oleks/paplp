#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cstddef>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <thrust/host_vector.h>

#include "types.hpp"
#include "problem.hpp"
#include "show.hpp"

#include "Matrix.hpp"
using paplp::Matrix;

using std::cout;
using std::endl;

using std::string;

texture<float, cudaTextureType2D, cudaReadModeElementType> textureRef;

#define BLOCK_SIZE 32

__global__ void interweavedReduction(char* blocks, Data* c, size_t N, size_t M)
{
  size_t blockId = blockIdx.y * blockDim.x + blockIdx.x;
  blocks[blockId] = 5;
}

__host__ int main()
{
  Data epsilon = 0.1;
  HostVector objective(3);
  HostVector constraintMatrix(9);
  HostVector constraintBounds(3);

  Problem initialProblem(
    epsilon,
    objective,
    constraintMatrix,
    constraintBounds);

  SpecialProblem specialProblem(initialProblem);

  ShowHostVector(specialProblem.objective);

  Matrix coefficients(100,100);

/*
  size_t N = 200;
  size_t BLOCK = 10;
  size_t NoOfBlocks = N / BLOCKS;
  size_t SIZE = sizeof(int) * N;

  int* hostValues = (data*)malloc(SIZE);

  int* deviceValues;
  cudaMalloc(&deviceValues, SIZE);

  char* blocks;
  cudaMalloc(&blocks, sizeof(char) * NoOfBlocks);

  cudaMemcpy(deviceValues, hostValues, SIZE, cudaMemcpyHostToDevice);
  exponent<<<N/BLOCK, BLOCK>>>(deviceValues);
  cudaMemcpy(hostValues, deviceValues, SIZE, cudaMemcpyDeviceToHost);

  return 0;
*/
}
