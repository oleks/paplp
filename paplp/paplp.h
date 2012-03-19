#ifndef PAPLP_H
#define PAPLP_H

#include <stdint.h>

#include "lib.h"

enum Form { PRIMAL, DUAL };
enum Type { POSITIVE, NORMALIZED };

typedef struct Problem
{
  uint32_t  noOfVariables;
  uint32_t  noOfConstraints;
  double    epsilon;
  double*   objectiveCoefficients;
  double*   constraintValues;
  double*   constraintCoefficients;
  Form      form;
  Type      type;
} Problem;

inline void transpose(Problem* problem)
{
  SWAP(problem->noOfVariables, problem->noOfConstraints);
}

#endif
