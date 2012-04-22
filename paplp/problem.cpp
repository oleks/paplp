#include <iostream>
using std::cout;
using std::endl;

#include "types.hpp"

#include "problem.hpp"

Data ApproximationProblem::Epsilon(void)
{
  return this->epsilon;
}

SpecialProblem::SpecialProblem(
  Data epsilon,
  size_t noOfVariables,
  size_t noOfContraints,
  Data *objective,
  Data *constraintMatrix)
  : ApproximationProblem(epsilon),
    noOfVariables(noOfVariables),
    noOfConstraints(noOfConstraints),
    objective(objective),
    constraintMatrix(constraintMatrix)
{
  // Intentionally left blank. See initialization list.
};


Problem::Problem(
  Data epsilon,
  size_t noOfVariables,
  size_t noOfContraints,
  Data *objective,
  Data *constraintMatrix,
  Data *constraintBounds)
  : ApproximationProblem(epsilon),
    noOfVariables(noOfVariables),
    noOfConstraints(noOfConstraints),
    objective(objective),
    constraintMatrix(constraintMatrix),
    constraintBounds(constraintBounds)
{
  // Intentionally left blank. See initialization list.
};

Problem::operator SpecialProblem()
{
  return SpecialProblem(
    this->epsilon,
    this->noOfVariables,
    this->noOfConstraints,
    this->objective,
    this->constraintMatrix);
};
