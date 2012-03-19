#include <float.h>
#include <math.h>
#include <stdio.h>

#include "lib.h"
#include "normalize.h"

inline double firstNormalization(Problem* problem);
inline void secondNormalization(Problem* problem, double beta);

void normalize(Problem* problem)
{
  secondNormalization(problem, firstNormalization(problem));

/* We need not set all b's to 1 as this will be superimposed by a change of the
 * problem type to NORMALIZED. */

  problem->type = NORMALIZED;
}

inline double firstNormalization(Problem* problem)
{
  double*  d = problem->objectiveCoefficients;
  double*  b = problem->constraintValues;
  double*  c = problem->constraintCoefficients;

  double beta = DBL_MAX;
  for (uint32_t j = 0; j < problem->noOfConstraints; ++j)
  {
    double beta_j = DBL_MIN;
    for (uint32_t i = 0; i < problem->noOfVariables; ++i)
    {
      *c = *c / ( b[j] * d[i] );
      printf("C: %e\n", *c);
      beta_j = MAX(beta_j, *c);
      c++;
    }
    printf("betaj: %e\n", beta_j);
    beta = MIN(beta, beta_j);
  }

  printf("beta: %e\n", beta);

  return beta;
}

#define GET_COEFFICIENTS() \
  double* c = problem->constraintCoefficients;\
  uint32_t noOfCoefficients = problem->noOfConstraints * problem->noOfVariables;

inline void secondNormalization(Problem *problem, double beta)
{
  GET_COEFFICIENTS();

  double upperBound = ( beta * problem->noOfConstraints ) / problem->epsilon;
  double lowerBound = ( problem->epsilon * beta ) / problem->noOfConstraints;

  printf("upper bound: %e; lower bound: %e\n", upperBound, lowerBound);

  for (uint32_t i = 0; i < noOfCoefficients; ++i, c++)
  {
    if(*c > upperBound)
    {
      *c = upperBound;
    }
    else if (*c < lowerBound)
    {
      *c = 0;
    }
  }
}

void checkNormalization(Problem* problem)
{
  GET_COEFFICIENTS();

  double t = DBL_MIN;
  double b = DBL_MAX;

  for (uint32_t i = 0; i < noOfCoefficients; ++i, c++)
  {
    t = MAX(t, *c);
    if (*c > 0)
    {
      b = MIN(b, *c);
      printf("c>0: %e\n",*c);
    }
  }

  double ratio = t / b;
  double gamma = pow(problem->noOfConstraints, 2) / pow(problem->epsilon, 2);

  printf("t/b vs. gamma = %e vs %e\n", ratio, gamma);
}

#undef GET_COEFFICIENTS
