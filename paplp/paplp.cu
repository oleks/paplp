#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <thrust/host_vector.h>

#include "paplp.h"
#include "normalize.h"

using std::cout;
using std::endl;

using std::string;

int main()
{
  double e = 0.1;
  uint32_t n = 3;
  uint32_t m = 3;
  double d[3] = { 7, 8, 9 };
  double b[3] = { 4, 5, 6 };
  double c[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

  Problem problem =
    {
      .noOfVariables = n,
      .noOfConstraints = m,
      .epsilon = e,
      .objectiveCoefficients = d,
      .constraintValues = b,
      .constraintCoefficients = c,
      .form = PRIMAL,
      .type = POSITIVE
    };

  normalize(&problem);
  checkNormalization(&problem);

  printf("%d\n", problem.noOfConstraints);
}


